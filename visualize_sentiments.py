# Project description:
# I want to take a text classification dataset and perform dimensionality reduction on it in two ways:
# 1. Take a pre-trained bert model and use UMAP to visualize the embeddings (easy)
# 2. Train parametric UMAP to visualize the embeddings - need to freeze bert layers beforehand (a bit harder)
# 3. Co-train parametric UMAP head and classification head (hard)
# The goal is to compare the resulting visualizations in each method. At first I will visually inspect the results and see which one looks the best
# but I think it will be interesting to use some kind of measure for visualization quality as well.


import random
from transformers import AutoTokenizer, TFAutoModel, TFBertTokenizer
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pandas as pd
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig
import hydra
import tensorflow as tf
import logging
import os
from sklearn.preprocessing import LabelEncoder
import umap
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set, find_ab_params
from sklearn.utils import check_random_state
from umap.parametric_umap import construct_edge_dataset
from umap.parametric_umap import umap_loss


HF_MODEL_NAMES = {'bert': 'bert-base-cased',
                  'sbert': 'sentence-transformers/all-MiniLM-L6-v2',
                  'simcse': 'princeton-nlp/sup-simcse-bert-base-uncased',
                  'bertmini': 'google/bert_uncased_L-4_H-256_A-4'}


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = tf.cast(
        tf.broadcast_to(tf.expand_dims(attention_mask, -1),
                        tf.shape(token_embeddings)),
        tf.float32
    )
    return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)


class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_name_or_path, mode, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model
        self.mode = mode
        try:
            self.model = TFAutoModel.from_pretrained(
                model_name_or_path, **kwargs)
        except OSError:
            self.model = TFAutoModel.from_pretrained(
                model_name_or_path, from_pt=True, **kwargs)

    def call(self, inputs, normalize=True):
        # runs model on inputs
        model_output = self.model(inputs)
        # Perform pooling. In this case, mean pooling.
        if self.mode == 'average':
            embeddings = mean_pooling(model_output, inputs["attention_mask"])
        elif self.mode == 'cls':
            embeddings = model_output[0][:, 0, :]
        else:
            raise ValueError('Invalid mode for SentenceTransformer')
        # normalizes the embeddings if wanted
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings

    def normalize(self, embeddings):
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings


class E2ESentenceTransformer(tf.keras.Model):
    def __init__(self, model_name_or_path, mode, **kwargs):
        super().__init__()
        # loads the in-graph tokenizer
        try:
            self.tokenizer = TFBertTokenizer.from_pretrained(
                model_name_or_path, **kwargs)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, **kwargs)
        # loads our TFSentenceTransformer
        self.model = TFSentenceTransformer(model_name_or_path, mode, **kwargs)

    def call(self, inputs):
        # runs tokenization and creates embedding
        tokenized = self.tokenizer(inputs)
        return self.model(tokenized)


class UMAPExtender(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.encoder = model
        self.encoder.trainable = False
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        (to_x, from_x) = inputs
        # parametric embedding
        embedding_to = self.encoder(to_x)
        embedding_from = self.encoder(from_x)
        viz_to = self.dense(embedding_to)
        viz_from = self.dense(embedding_from)

        # concatenate to/from projections for loss computation
        viz_to_from = tf.concat([viz_to, viz_from], axis=1)
        viz_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(
            viz_to_from
        )
        outputs = {'umap': viz_to_from}
        return outputs


class UMAPNormalModel(tf.keras.Model):
    def __init__(self, encoder, dense_layer):
        super().__init__()
        self.encoder = encoder
        self.dense = dense_layer

    def call(self, inputs):
        embedding = self.encoder(inputs)
        viz = self.dense(embedding)
        return viz


class CustomParametricUMAP:
    def __init__(self, raw_data, data_for_neighbor_graph, bert_model, batch_size=32, n_epochs=1):
        self.raw_data = raw_data
        self.data_for_neighbor_graph = data_for_neighbor_graph
        self.bert_model = bert_model
        self.n_neighbors = 10
        self.metric = "cosine"
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def get_neighbor_graph(self):
        X = self.data_for_neighbor_graph
        # number of trees in random projection forest
        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        # get nearest neighbors
        nnd = NNDescent(
            X.reshape((len(X), np.product(np.shape(X)[1:]))),
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        # get indices and distances
        return nnd.neighbor_graph

    def get_fuzzy_simplicial_set(self):
        knn_indices, knn_dists = self.get_neighbor_graph()
        random_state = check_random_state(None)
        umap_graph, _, _ = fuzzy_simplicial_set(
            X=self.data_for_neighbor_graph,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return umap_graph

    def get_edge_dataset(self):
        umap_graph = self.get_fuzzy_simplicial_set()
        n_epochs = self.n_epochs
        (
            edge_dataset,
            batch_size,
            n_edges,
            head,
            tail,
            edge_weight,
        ) = construct_edge_dataset(
            self.raw_data,
            umap_graph,
            n_epochs,
            self.batch_size,
            parametric_embedding=True,
            parametric_reconstruction=False,
            global_correlation_loss_weight=0
        )
        return edge_dataset, n_edges, edge_weight

    def fit_transform(self, X):
        edge_dataset, n_edges, edge_weight = self.get_edge_dataset()
        parametric_model = UMAPExtender(self.bert_model)
        # create model
        optimizer = tf.keras.optimizers.Adam(1e-3)
        min_dist = 0.1
        _a, _b = find_ab_params(1.0, min_dist)
        negative_sample_rate = 5
        umap_loss_fn = umap_loss(
            self.batch_size,
            negative_sample_rate,
            _a,
            _b,
            edge_weight,
            parametric_embedding=True
        )
        parametric_model.compile(
            optimizer=optimizer, loss=umap_loss_fn, run_eagerly=True
        )
        steps_per_epoch = int(
            n_edges / self.batch_size / 5
        )
        history = parametric_model.fit(
            edge_dataset,
            epochs=self.n_epochs,
            steps_per_epoch=steps_per_epoch,
            max_queue_size=100,
        )
        normal_model = UMAPNormalModel(
            parametric_model.encoder, parametric_model.dense)
        return normal_model.predict(X)


def preprocess_data_df(df, sampling_ratio=1):
    return (df
            .drop_duplicates(subset='SentenceId', keep='first')
            .sample(frac=sampling_ratio, axis=1)
            )


class TextEmbedder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name)
        self.dataset = None
        self.classification_model = None
        self.label_encoder = None
        self.data_df = None
        self.api = KaggleApi()
        self.api.authenticate()

    def get_embeddings_model(self, aggregation_method):
        bert = TFAutoModel.from_pretrained(self.cfg.bert_model_name)
        input_ids = tf.keras.layers.Input(
            shape=(50,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(
            shape=(50,), name='attention_mask', dtype='int32')
        embeddings = bert(input_ids, attention_mask=mask)[0]
        if aggregation_method == 'average':
            average_embeddings = mean_pooling(embeddings, mask)
            # average_embeddings = tf.math.reduce_mean(embeddings, axis=1)
        elif aggregation_method == 'cls':
            average_embeddings = embeddings[:, 0, :]
        else:
            raise Exception("Invalid aggregation method for bert")
        model = tf.keras.Model(
            inputs=[input_ids, mask], outputs=average_embeddings)
        return model

    def get_bert_model(self):
        if self.classification_model is not None:
            return self.classification_model
        else:
            bert = TFAutoModel.from_pretrained(self.cfg.bert_model_name)
            input_ids = tf.keras.layers.Input(
                shape=(50,), name='input_ids', dtype='int32')
            mask = tf.keras.layers.Input(
                shape=(50,), name='attention_mask', dtype='int32')
            embeddings = bert(input_ids, attention_mask=mask)[0]
            X = tf.keras.layers.LSTM(64)(embeddings)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Dense(64, activation='relu')(X)
            X = tf.keras.layers.Dropout(0.1)(X)
            y = tf.keras.layers.Dense(
                self.cfg.n_classes, activation='softmax', name='outputs')(X)
            model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
            model.layers[2].trainable = False
            self.classification_model = model
            return model

    def get_model(self, model_name, mode):
        self.model = E2ESentenceTransformer(model_name, mode=mode)
        return self.model

    def get_data(self):
        if self.dataset is not None:
            return self.dataset
        else:
            logging.info(f'Acquiring dataset: {self.cfg.dataset}...')
            self.dataset = {}
            if self.cfg.dataset == 'news':
                dataset = self.get_news_dataset()
            elif self.cfg.dataset == 'sentiment':
                dataset = self.get_sentiment_dataset()
            dataset_size = len(dataset)
            self.dataset['train'] = dataset.take(
                int((1 - self.cfg.validation_set_size) * dataset_size))
            self.dataset['validation'] = dataset.take(
                int(self.cfg.validation_set_size * dataset_size))
            self.dataset['full'] = dataset
            return self.dataset

    def get_sentiment_dataset(self):
        self.api.competition_download_file(
            'sentiment-analysis-on-movie-reviews', 'train.tsv.zip', path=f'{self.cfg.dataset_dir}/sentiment')
        with zipfile.ZipFile(f'{self.cfg.dataset_dir}/sentiment/train.tsv.zip', 'r') as z:
            z.extractall(f'{self.cfg.dataset_dir}/sentiment')
        dataset = {}
        self.data_df = (pd.read_csv(f'{self.cfg.dataset_dir}/sentiment/train.tsv.zip', sep='\t')
                        .drop_duplicates(subset='SentenceId', keep='first')
                        .sample(frac=self.cfg.data_sampling_ratio, axis=1))
        encoded_sentences = self.encode_sentences(
            self.data_df.Phrase.values.tolist())
        logging.info(
            f'Unique sentiment values in train set: {pd.unique(self.data_df.Sentiment)}')
        dataset = self.create_tf_dataset(encoded_sentences['input_ids'],
                                         encoded_sentences['attention_mask'], self.data_df.Sentiment.values)
        return dataset

    def get_news_dataset(self):
        filename = 'News_Category_Dataset_v3.json'
        self.api.dataset_download_file(
            'rmisra/news-category-dataset', filename, path=f'{self.cfg.dataset_dir}/news')
        with zipfile.ZipFile(f'{self.cfg.dataset_dir}/news/{filename}.zip', 'r') as z:
            z.extractall(f'{self.cfg.dataset_dir}/news')
        self.data_df = (pd.read_json(os.path.join(self.cfg.dataset_dir, 'news', filename), lines=True)
                        .assign(long_description=lambda df: df.headline + ' ' + df.short_description)
                        .query('category in ["SPORTS", "FOOD & DRINK", "STYLE", "WORLDPOST", "DIVORCE"]')
                        .sample(frac=self.cfg.data_sampling_ratio, axis=0))
        encoded_articles = self.encode_sentences(
            self.data_df.long_description.values.tolist())
        logging.info(
            f'Unique categories values in dataset: {pd.unique(self.data_df.category)}')
        dataset = self.create_tf_dataset(encoded_articles['input_ids'],
                                         encoded_articles['attention_mask'], self.data_df.category.values)
        return dataset

    def tokenize(self, sentence):
        tokens = self.tokenizer.encode_plus(sentence, max_length=self.cfg.seq_len,
                                            truncation=True, padding='max_length',
                                            add_special_tokens=True, return_attention_mask=True,
                                            return_token_type_ids=False, return_tensors='tf')
        return tokens['input_ids'], tokens['attention_mask']

    def encode_sentences(self, sentences):
        return self.tokenizer.batch_encode_plus(sentences, max_length=self.cfg.seq_len, truncation=True, padding='max_length',
                                                add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False, return_tensors='tf')

    def create_tf_dataset(self, input_ids, masks, labels):
        def map_func(input_ids, masks, labels):
            return {'input_ids': input_ids, 'attention_mask': masks}, labels
        if type(labels[0]) is str:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(labels)
        labels_transformed = tf.constant(tf.keras.utils.to_categorical(labels))
        return (tf.data.Dataset.from_tensor_slices((input_ids, masks, labels_transformed))
                .map(map_func)
                .shuffle(100000)
                .batch(self.cfg.batch_size)
                )

    def train_classification_model(self):
        self.get_data()
        self.get_bert_model()
        optimizer = tf.keras.optimizers.Adam(0.01)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
        self.classification_model.compile(
            optimizer=optimizer, loss=loss, metrics=[acc])
        self.history = self.classification_model.fit(
            self.dataset['train'], validation_data=self.dataset['validation'], epochs=40)

    def get_representations_and_labels(self, representation_method):
        self.get_data()
        representation_class = representation_method.split('_')[0]
        if representation_class == 'tfidf':
            tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
            representations = tfidf_vectorizer.fit_transform(
                self.data_df.long_description.values)
            labels = self.data_df.category.values
        elif representation_class in list(HF_MODEL_NAMES.keys()):
            model_name = HF_MODEL_NAMES[representation_class]
            model = self.get_model(
                model_name, mode=representation_method.split('_')[1])
            representations = model.predict(
                self.data_df.long_description.values.tolist(), batch_size=32)
            labels = self.data_df.category.values
        else:
            raise Exception(
                f'Invalid representation method: {representation_method}')
        return representations, labels

    def visualize_embeddings_umap(self):
        all_res = []
        for representation_method in self.cfg.representation_methods:
            res = {}
            logging.info(
                f'calculating representations for {representation_method}')
            representations, labels = self.get_representations_and_labels(
                representation_method)
            if '_parametric' in representation_method:
                reducer = umap.parametric_umap.ParametricUMAP()
                embedding = reducer.fit_transform(representations)
            elif '_customparametric' in representation_method:
                reducer = CustomParametricUMAP(raw_data=self.data_df.long_description.values,
                                               data_for_neighbor_graph=representations,
                                               bert_model=self.model)
                embedding = reducer.fit_transform(
                    self.data_df.long_description.values)
            else:
                reducer = umap.UMAP(metric='cosine')
                embedding = reducer.fit_transform(representations)
            fig = px.scatter(x=embedding[:, 0],
                             y=embedding[:, 1], color=labels)
            viz_filename = f'{representation_method}_embedding_viz.html'
            fig.write_html(viz_filename)
            res['method'] = representation_method
            res['path'] = os.path.join(os.getcwd(), viz_filename)
            res['embedding'] = embedding
            res['labels'] = labels
            all_res.append(res)
        return all_res


def calculate_distance_consistency(viz_result):
    def get_closest_centroid(row, centroids):
        min_dist = np.inf
        row_x_y = row[['x', 'y']].values
        for _, cent_row in centroids.iterrows():
            dist = np.linalg.norm(row_x_y - cent_row[['x', 'y']])
            if dist < min_dist:
                closest_centroid = cent_row.name
                min_dist = dist
        return closest_centroid

    viz_df = pd.DataFrame({'labels': viz_result['labels'],
                           'x': viz_result['embedding'][:, 0],
                           'y': viz_result['embedding'][:, 1]})
    centroids = viz_df.groupby('labels').mean()
    viz_df_with_closest_centroids = viz_df.assign(closest_centroid=viz_df.apply(
        lambda row: get_closest_centroid(row, centroids), axis=1),
        correct_closest_centroid=lambda df: df.closest_centroid != df.labels)
    return viz_df_with_closest_centroids.correct_closest_centroid.sum() / len(viz_df_with_closest_centroids)


def run_training_loop(cfg):
    tf.random.set_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    text_embedder = TextEmbedder(cfg)
    if cfg.task == 'train_classification_model':
        text_embedder.train_classification_model()
    elif cfg.task == 'visualize_embeddings':
        viz_results = text_embedder.visualize_embeddings_umap()
        for viz_result in viz_results:
            viz_result['consistency'] = calculate_distance_consistency(
                viz_result)
        consistency_barplot = px.bar(x=[res['method'] for res in viz_results],
                                     y=[res['consistency']
                                         for res in viz_results],
                                     title='Distance consistency of UMAP per embedding method')
        consistency_barplot.write_html('consistency_comparison.html')


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    run_training_loop(cfg)


# def main() -> None:
#     with initialize(config_path="drive/MyDrive/code/ParametricUMAP"):
#         cfg=compose(config_name="config.yaml")
#         run_training_loop(cfg)


if __name__ == "__main__":
    main()
