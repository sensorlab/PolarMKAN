from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.metrics import get_supervised_metrics_features
import tqdm
import pickle
import math
import src.metrics as metrics


class Trainer(ABC):
    """
    Abstract base class for training models.
    
    This class defines the interface that all trainer classes should implement.
    """

    @abstractmethod
    def train_epoch():
        """
        Train the model for one epoch.
        """
        pass

    @abstractmethod
    def get_features():
        """
        Get final embeddings of the data.
        """
        pass

    @abstractmethod
    def save_checkpoint():
        """
        Save a checkpoint of the model.
        """
        pass

    @abstractmethod
    def load_checkpoint():
        """
        Load a checkpoint of the model.
        """
        pass

    @abstractmethod
    def evaluate():
        """
        Evaluate the model.
        """
        pass


class SIM_CLR_Trainer(Trainer):
    """
    Trainer class for contrastive learning using a SimCLR-like approach.

    Inspired by:
        - X. Hao et al., "Contrastive self-supervised clustering for specific emitter identification," IEEE IoT Journal, 2023.
        - "Viewmaker Networks: Learning Views for Unsupervised Representation Learning", Alex Tamkin et al.

    Improvements include "large_augs" which apply augmentations inside feature extractor layers.

    Dataset used with DataLoader must support the "return_indices" flag.
    Feature extractor must return two outputs: (features, auxiliary_output).
    """

    def __init__(
        self,
        models: nn.ModuleDict,
        optimizers: dict,
        hard_postitves_mining: bool = False,
        hard_negatives_mining: bool = False,
        temperature: float = 1,
        clusters_loss: bool = False,
        augs_type="static",
        num_epochs=200,
        device="cpu",
        large_augs=False,
    ):
        """
        Init the ContrastiveTrainer object.

        Args:
            models (nn.ModuleDict): Dict of models used in training. Format: {
                'features extrcactor': features extractor model,
                'mlp_instance': mlp head for instance loss,
                'mlp_cluster': mlp head for cluster loss, optional,
                'augs': augmentations}.
            hard_postitves_mining (bool): remove 1/4 positives with the hieghst similarity, defalut False.
            hard_negatives_mining (bool): remove 1/4 negatives with the lowest similarity, defalut False.
            temperature (float): temperature in cross entropy loss.
            clusters_loss (bool): to use cluster loss or not.
            augs_type (str): 'static' or 'learnable'.
            total_epochs (int): num epochs to learn.
            optimizers (dict): Optimizers for learning in format: {
                'main_optimizer': optimzer for features extractors and mlp heads,
                'augs_optimizer': optimzer for augs,
                }
        """
        self.hard_postitves_mining = hard_postitves_mining
        self.hard_negatives_mining = hard_negatives_mining
        self.models = models
        self.optimizers = optimizers
        self.augs_type = augs_type
        self.clusters_loss = clusters_loss
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.device = device
        self.large_augs = large_augs
        self.epoch = 0


    def svd_init(self,  model, train_loader: torch.utils.data.DataLoader):
        inputs = []
        for input, _ in train_loader:
            inputs.append(input)
        inputs = torch.cat(inputs)
        inputs = inputs.reshape(inputs.shape[0],-1)
        pca = PCA(20)
        pca.fit(inputs)
        pca_components = pca.components_
        pca_weights_tensor = torch.from_numpy(pca_components)
        with torch.no_grad():
            model.svd_init_layer.weight.data.copy_(pca_weights_tensor)
            model.svd_init_layer.bias.data.zero_()

    def similarity(self, first, second, type="cosine"):
        """
        Compute similarity matrix between two batches of embeddings.

        Args:
            first (torch.Tensor): Tensor of shape (batch_size, features).
            second (torch.Tensor): Tensor of shape (batch_size, features).
            type (str): Type of similarity - 'cosine' or 'l_2'.

        Returns:
            torch.Tensor: Similarity matrix.
        """
        if type == "cosine":
            first = first / torch.norm(first, dim=1, keepdim=True)
            second = second / torch.norm(second, dim=1, keepdim=True)
            return first @ second.T

        if type == "l_2":
            return torch.norm(first[:, None, :] - second[None, :, :], dim=-1)

    def compute_loss(
        self, p_samples: torch.Tensor, q_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between pairs of positive samples.

        Args:
            p_samples (torch.Tensor): Positive sample features.
            q_samples (torch.Tensor): Corresponding query sample features.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        loss_f = torch.nn.CrossEntropyLoss()

        # Compute similarity matrices, (batch_size, batch_size)
        p_p_sim = self.similarity(p_samples, p_samples, type="cosine")
        q_q_sim = self.similarity(q_samples, q_samples, type="cosine")
        p_q_sim = self.similarity(p_samples, q_samples, type="cosine")

        # Remove the diagonal elements from similarity matrices, (batch_size, batch_size-1)
        n = p_samples.size(0)
        mask = torch.eye(n, device=p_p_sim.device, dtype=torch.bool)
        p_p_sim = p_p_sim[~mask].view(n, n - 1)
        q_q_sim = q_q_sim[~mask].view(n, n - 1)

        if self.hard_negatives_mining:
            # Sort and select top 75% similar negatives
            p_p_sim, _ = torch.sort(p_p_sim, dim=-1, descending=True)
            p_p_sim = p_p_sim[:, : (3 * p_p_sim.shape[1]) // 4]

            q_q_sim, _ = torch.sort(q_q_sim, dim=-1, descending=True)
            q_q_sim = q_q_sim[:, : (3 * q_q_sim.shape[1]) // 4]

        # Concatenate similarities for cross-entropy loss
        p_sims = torch.cat([p_q_sim, p_p_sim], 1)
        q_sims = torch.cat([p_q_sim, q_q_sim], 1)

        labels = torch.arange(p_sims.shape[0], device=p_sims.device, dtype=torch.long)

        if self.hard_postitves_mining:
            # Sort and select top 75% disimilar positives
            _, indices = torch.sort(torch.diag(p_q_sim), descending=True)
            indices = indices[len(indices) // 4 :]

            p_sims = p_sims[indices]
            q_sims = q_sims[indices]
            labels = labels[indices]

        # Compute loss
        l_p = loss_f(p_sims / self.temperature, labels)
        l_q = loss_f(q_sims / self.temperature, labels)

        loss = torch.mean(l_p + l_q) / 2

        return loss

    def _train_step(self, batch: torch.Tensor, type: str="features extractor"):
        """
        Process a batch through the model pipeline.

        Args:
            batch (torch.Tensor): Input batch.
            type (str): Type of training ('features extractor' or 'augs').

        Returns:
            float: Loss value.
        """
        batch = batch.to(self.device)

        features_extractor = self.models["feature_extractor"].to(self.device)
        features_extractor.train()

        augs = self.models["augs"].to(self.device)
        augs.train()

        if self.clusters_loss:
            mlp_cluster = self.models["mlp_cluster"].to(self.device)
            mlp_cluster.train()

        mlp_instance = self.models["mlp_instance"].to(self.device)
        mlp_instance.train()
            
        p_aug, q_aug = np.random.choice(augs, size=2)
        batch_p = p_aug(batch)
        batch_q = q_aug(batch)
        
        p_features, _ = features_extractor(batch_p)
        q_features, _ = features_extractor(batch_q)

        p_instance, q_instance = mlp_instance(p_features), mlp_instance(q_features)
        loss = self.compute_loss(p_instance, q_instance)

        if self.clusters_loss:
            p_clusters, q_clusters = mlp_instance(mlp_cluster), mlp_instance(
                mlp_cluster
            )
            loss += self.compute_loss(p_clusters.T, q_clusters.T)

        if type == "features extractor":
            optimizer = self.optimizers["main_optimizer"]
            
        elif type == 'augs':
            loss = -loss
            optimizer = self.optimizers["augs_optimizer"]
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader, scheduler=None
    ) -> float:
        """
        Train the feature extractor for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            scheduler (optional): Learning rate scheduler. Default is None.

        Returns:
            float: Loss for the epoch.
        """
        assert hasattr(train_loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"
            
        running_loss = 0
        c = 0


        if self.epoch == 0:
            self.svd_init(self.models['feature_extractor'], train_loader)
            self.epoch += 1

        train_loader.dataset.return_indices = False

        for batch_inputs, batch_device_ids in train_loader:
            loss = self._train_step(batch_inputs, "features extractor")

            if self.augs_type == "learnable":
                loss -= self._train_step(batch_inputs, "augs")
            
            running_loss += loss
            c += 1

        if scheduler:
            scheduler.step()

        return running_loss / c

    def get_features(
        self, loader: torch.utils.data.DataLoader, type: str="features"
    ) -> torch.Tensor:
        """
        Get final feature embeddings from the model.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader with input data.
            type (str): 'featutres' if return embeding, 'scores' if return softmax scores

        Returns:
            torch.Tensor: Concatenated features from all batches.
        """
        assert hasattr(loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"
            
        model = self.models["feature_extractor"].to(self.device)
        model.eval()

        if self.clusters_loss and type == "scores":
            mlp_cluster = self.models["mlp_cluster"].to(self.device)
            mlp_cluster.eval()

        all_features = []

        loader.dataset.return_indices = False

        with torch.no_grad():
            for inputs, target in loader:
                inputs, target = inputs.to(self.device), target.to(self.device)

                features, _ = model(inputs)

                if self.clusters_loss and type == "probas":
                    features = mlp_cluster(features)

                all_features.append(features.cpu())

        return torch.cat(all_features)

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        """
        Evaluate model using clustering and supervised metrics.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
            targets (torch.Tensor): Binary tensor indicating known (0) vs unknown (1) classes.
            clusters_numbers (tuple): Number of clusters to evaluate.

        Returns:
            dict: Computed evaluation metrics.
        """
        train_features = self.get_features(train_loader)
        test_features = self.get_features(test_loader)

        train_loader.dataset.return_indices = False
        test_loader.dataset.return_indices = False

        train_devices = [devices for _, devices in train_loader]
        test_devices = [devices for _, devices in test_loader]

        train_devices = torch.cat(train_devices).numpy()
        test_devices = torch.cat(test_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers, test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers
        )
        
        if self.clusters_loss:
            train_probas = self.get_features(train_loader, type="scores")
            test_probas = self.get_features(test_loader, type="scores")
            
            supervised_metrics_probas = metrics.get_supervised_metrics_probas(
                train_probas, test_probas, targets, clusters_numbers
            )

        all_metics = (
            supervised_metrics_features
            | unsupervised_metrics_features
        )
        
        if self.clusters_loss:
            all_metics |= supervised_metrics_probas

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        checkpoint = {"state_dict": self.models.state_dict()}
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path)
        self.models.load_state_dict(checkpoint["state_dict"])


class AE_Trainer(Trainer):
    """
    Classical Auto Encoder.

    Inspired by L. Milosheski, M. Mohorčič and C. Fortuna, "Spectrum Sensing With Deep Clustering: Label-Free Radio Access Technology Recognition,"
    in IEEE Open Journal of the Communications Society, vol. 5, pp. 4746-4763, 2024, doi: 10.1109/OJCOMS.2024.3436601.
    """

    def __init__(
        self,
        models: nn.ModuleDict,
        optimizers: dict,
        noise_std: float = 0,
        num_epochs=200,
        device="cuda",
        distance_loss=False,
        clusters_update_interval=4,
        n_clusters=40,
    ):
        """
        Args:
            noise_std (float): Amount of normal noise applied to the signal.
            models (nn.ModuleDict): Dict of models used in training. Format: {
                'features_extractor': features extractor model}.
            optimizers (dict): {'main_optimizer': optimizer for features extractor}.
            num_epochs (int): Number of epochs.
            device (str): 'cuda' or 'cpu'.
            distance_loss (bool): Apply distance loss or not.
            clusters_update_interval (int): Interval between cluster updates.
        """
        self.noise_std = noise_std
        self.models = models
        self.optimizers = optimizers
        self.num_epochs = num_epochs
        self.device = device
        self.clusters_centers = None
        self.clusters_update_interval = clusters_update_interval
        self.n_clusters = n_clusters
        self.distance_loss = distance_loss
        self.cur_counter = 0
        self.epoch = 0

    def svd_init(self,  model, train_loader: torch.utils.data.DataLoader):
        inputs = []
        for input, _ in train_loader:
            inputs.append(input)
        inputs = torch.cat(inputs)
        inputs = inputs.reshape(inputs.shape[0],-1)
        pca = PCA(20)
        pca.fit(inputs)
        pca_components = pca.components_
        pca_weights_tensor = torch.from_numpy(pca_components)
        with torch.no_grad():
            model.svd_init_layer.weight.data.copy_(pca_weights_tensor)
            model.svd_init_layer.bias.data.zero_()
        

    def _update_clusters(self, train_loader: torch.utils.data.DataLoader):
        """
        Update clusters using KMeans clustering on feature embeddings.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
        """
        train_features = self.get_features(train_loader)

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(train_features)

        self.clusters_centers = torch.tensor(kmeans.cluster_centers_)

    def calc_distance_loss(self, features: torch.tensor)->torch.Tensor:
        """
        Calculate distance-based loss as defined in:
        H. Zhou et al., “Deep radio signal clustering with interpretability analysis based on saliency map", 2023.

        Args:
            features (torch.tensor): Features of shape (batch_size, features_size).
        
        Returns:
            torch.tensor: Distance loss.
        """
        num_clusters = self.clusters_centers.shape[0]
        num_samples = features.shape[0]

        # Expand features and cluster centers for distance calculation
        features = features.unsqueeze(0).repeat(num_clusters, 1, 1)
        clusters_centres = (
            self.clusters_centers.unsqueeze(1)
            .repeat(1, num_samples, 1)
            .to(features.device)
        )

        distances = ((features - clusters_centres) ** 2).sum(axis=2)
        
        # Compute weighted distance loss
        min_distances, _ = distances.detach().min(dim=0)
        min_distances = min_distances.unsqueeze(0).repeat(num_clusters, 1)

        exp_shifted_distances = torch.exp(-(distances - min_distances)).detach()
        exp_shifted_distances_sums = exp_shifted_distances.sum(axis=0)

        weighted_distances = (
            distances * exp_shifted_distances / exp_shifted_distances_sums
        )

        return weighted_distances.mean()

    def train_epoch(self, train_loader, scheduler=None) -> float:
        """
        Train the feature extractor for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            scheduler (optional): Learning rate scheduler. Default is None.

        Returns:
            float: Loss for the epoch.
        """
        assert hasattr(train_loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"
        
        model = self.models["feature_extractor"].to(self.device)
        
        if self.epoch == 0:
            #self.svd_init(model, train_loader)
            self.epoch += 1

        model.train()

        optimizer = self.optimizers["main_optimizer"]
        running_loss = 0
        c = 0
        train_loader.dataset.return_indices = False

        if self.distance_loss and self.cur_counter % self.clusters_update_interval == 0:
            self._update_clusters(train_loader)
            self.cur_counter = 0

        self.cur_counter += 1

        for inputs, target in train_loader:
            inputs, target = inputs.to(self.device), target.to(self.device)
            x = (
                inputs
                + torch.randn(inputs.shape, device=inputs.device) * self.noise_std
            ) / (1 + self.noise_std)

            features, reconstructed = model(x)
            b_s = features.shape[0]
            features = features.view(b_s, -1)

            loss = torch.mean(
                (reconstructed.reshape(b_s, -1) - inputs.reshape(b_s, -1)) ** 2
            )

            if self.distance_loss:
                loss += self.calc_distance_loss(features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            c += 1

        if scheduler:
            scheduler.step()

        return running_loss / c

    def get_features(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get final feature embeddings from the model.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader with input data.

        Returns:
            torch.Tensor: Concatenated features from all batches.
        """
        assert hasattr(loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"
            
        model = self.models["feature_extractor"].to(self.device)
        model.eval()

        all_features = []
        loader.dataset.return_indices = False

        for inputs, target in loader:
            inputs, target = inputs.to(self.device), target.to(self.device)
            features, _ = model(inputs)
            all_features.append(features.detach().cpu().view(features.shape[0], -1))

        return torch.cat(all_features)

    def evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        targets: torch.tensor,
        clusters_numbers: tuple = (40,),
    ):
        """
        Calculate both supervised and unsupervised metrics.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            test_loader (torch.utils.data.DataLoader): DataLoader with testing data.
            targets (torch.tensor): List of targets (1 if unknown, 0 if known).
            clusters_numbers (tuple): Tuple of numbers of clusters to calculate metrics.

        Returns:
            dict: Combined metrics from both supervised and unsupervised approaches.
        """
        train_features = self.get_features(train_loader)
        test_features = self.get_features(test_loader)

        train_loader.dataset.return_indices = False
        test_loader.dataset.return_indices = False

        train_devices = [devices for _, devices in train_loader]
        test_devices = [devices for _, devices in test_loader]

        train_devices = torch.cat(train_devices).numpy()
        test_devices = torch.cat(test_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers, test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers
        )

        all_metics = (
            supervised_metrics_features
            | unsupervised_metrics_features
        )

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        torch.save(
            {
                "model_state_dict": self.models.state_dict(),
            },
            file_path,
        )

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path, weights_only=True)
        self.models.load_state_dict(checkpoint["model_state_dict"])


class PCA_Trainer(Trainer):

    def __init__(self, **approach_cfg):
        """
        Makes PCA

        Args:
            pca: PCA solver from as in sklearn
        """
        features_size = approach_cfg['features_size']
        self.pca = PCA(features_size)
        self.num_epochs = 5

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train the PCA extractor.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
                The flag "return_indices" must exist in train_loader.dataset object,
                which makes DataLoader return sample indices at the end of the tuple: (inputs, device_ids, ids).

        Returns:
            float: Explained variance ratio.
        """
        model = self.pca
        features_train = []
        train_loader.dataset.return_indices = False

        for inputs, _ in train_loader:
            features_train.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        features_train = torch.cat(features_train)
        model.fit(features_train)

        return model.explained_variance_ratio_.sum()

    def get_features(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader.

        Returns:
            torch.Tensor: Transformed features.
        """
        model = self.pca
        all_features = []
        loader.dataset.return_indices = False

        for inputs, _ in loader:
            all_features.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

        all_features = torch.cat(all_features)

        return torch.tensor(model.transform(all_features))

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)) -> dict:
        """
        Calculate supervised and unsupervised evaluation metrics.

        Args:
            train_loader (torch.utils.data.DataLoader): Training DataLoader.
            test_loader (torch.utils.data.DataLoader): Testing DataLoader.
            targets (torch.Tensor): Target list (1 for unknown, 0 for known).
            clusters_numbers (tuple): Cluster numbers for evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        train_features = self.get_features(train_loader)
        test_features = self.get_features(test_loader)

        train_loader.dataset.return_indices = False
        test_loader.dataset.return_indices = False

        train_devices = [devices for _, devices in train_loader]
        test_devices = [devices for _, devices in test_loader]

        train_devices = torch.cat(train_devices).numpy()
        test_devices = torch.cat(test_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers, test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers
        )

        all_metics = (
            supervised_metrics_features
            | unsupervised_metrics_features
        )

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the PCA extractor.

        Args:
            pca_extractor (PCA): PCA extractor from sklearn.
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        model = self.pca
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    def load_checkpoint(self, file_path: str) -> PCA:
        """
        Load a checkpoint of the PCA extractor.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            PCA: Loaded PCA extractor.
        """
        with open(file_path, "rb") as f:
            self.pca = pickle.load(f)


class Deep_Clustering_Trainer(Trainer):
    """
    Implementation of deep clustering algorithm.

    Based on: Caron, Mathilde, et al. "Deep clustering for unsupervised learning of visual features." 
    Proceedings of the European Conference on Computer Vision (ECCV), 2018.

    Note:
        Dataloader's dataset must have the attribute 'return_indices' set to True to return sample indices:
        (inputs, device_ids, ids).
    """

    def __init__(
        self,
        models,
        optimizers,
        clusters_update_interval: int = 5,
        n_clusters: int = 10,
        num_epochs=200,
        device="cuda",
    ):
        """
        Args:
            models (nn.ModuleDict): Models used for training. Format:
                {'feature_extractor': feature extractor model}
            optimizers (dict): Format:
                {'main_optimizer': optimizer for feature extractor}
            clusters_update_interval (int): Epochs between cluster updates.
            n_clusters (int): Number of clusters.
            num_epochs (int): Number of training epochs.
            device (str): Device to use ("cuda" or "cpu").
        """
        self.num_epochs = num_epochs
        self.device = device
        self.models = models
        self.optimizers = optimizers
        self.cur_counter = 0
        self.clusters_update_interval = clusters_update_interval
        self.loss_func = nn.CrossEntropyLoss()
        self.n_clusters = n_clusters
        self.softmax = nn.Softmax()
        self.p_labels = None
        self.epoch = 0

    def get_features(
        self,
        loader: torch.utils.data.DataLoader,
        type: str = "features",
        return_indices: bool = False,
    ) -> torch.Tensor:
        """
        Get final embeddings.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader.
            device (torch.device): Device to use (cpu or cuda).
            type (str): Type of output ('features' or 'scores').
            return_indices (bool): Whether to return indices.

        Returns:
            torch.Tensor: Features or scores.
        """
        assert hasattr(loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"
            
        model = self.models["feature_extractor"].to(self.device)
        model.eval()

        features_list = []
        ids_list = []

        loader.dataset.return_indices = True

        with torch.no_grad():
            for inputs, _, ids in loader:
                inputs = inputs.to(self.device)

                features, scores = model(inputs)

                if type == "features":
                    features_list.append(features.cpu())
                elif type == "scores":
                    features_list.append(self.softmax(scores.cpu()))

                ids_list.append(ids)

        if return_indices:
            return torch.cat(features_list), torch.cat(ids_list)
        else:
            return torch.cat(features_list)

    def _update_labels(self, train_loader: torch.utils.data.DataLoader):
        """
        Recalculate pseudo-labels using KMeans on PCA-reduced features.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader.

        """
        model = self.models["feature_extractor"].to(self.device)
        model.eval()

        train_features, ids = self.get_features(train_loader, return_indices=True)
        
        _, indices = torch.sort(ids,  descending=False)

        pca = PCA(20)
        train_features_reduced = pca.fit_transform(train_features)

        kmeans = KMeans(self.n_clusters)
        self.p_labels = torch.tensor(
            kmeans.fit_predict(train_features_reduced), dtype=torch.long
        )[indices]

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        scheduler=None,
    ) -> float:
        """
        Train the feature extractor for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            scheduler (optional): Learning rate scheduler. Default is None.

        Returns:
            float: Loss for the epoch.
        """
        assert hasattr(train_loader.dataset, 'return_indices'), \
            "'train_loader.dataset' must have a 'return_indices' attribute"

        running_loss = 0
        c = 0
        self.epoch += 1

        model = self.models["feature_extractor"].to(self.device)
        model.train()

        if self.cur_counter % self.clusters_update_interval == 0:
            self.cur_counter = 0
            self._update_labels(train_loader)

        self.cur_counter += 1
        train_loader.dataset.return_indices = True
        optimizer = self.optimizers["main_optimizer"]

        for inputs, _, ids in train_loader:
            inputs = inputs.to(self.device)
            p_labels = self.p_labels[ids].to(self.device)

            _, scores = model(inputs)
            loss = self.loss_func(scores, p_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            c += 1

        if scheduler:
            scheduler.step()

        return running_loss / c

    def evaluate(self, train_loader, test_loader, targets, clusters_numbers=(40,)):
        """
        Calculate supervised and unsupervised evaluation metrics.

        Args:
            train_loader (torch.utils.data.DataLoader): Training DataLoader.
            test_loader (torch.utils.data.DataLoader): Testing DataLoader.
            targets (torch.Tensor): Target list (1 for unknown, 0 for known).
            clusters_numbers (tuple): Cluster numbers for evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        train_features = self.get_features(train_loader)
        test_features = self.get_features(test_loader)

        train_loader.dataset.return_indices = False
        test_loader.dataset.return_indices = False

        train_devices = [devices for _, devices in train_loader]
        test_devices = [devices for _, devices in test_loader]

        train_devices = torch.cat(train_devices).numpy()
        test_devices = torch.cat(test_devices).numpy()

        supervised_metrics_features = metrics.get_supervised_metrics_features(
            train_features, test_features, targets, clusters_numbers, test_devices, train_devices
        )

        unsupervised_metrics_features = metrics.get_unsupervised_metrics_features(
            train_features, test_features, clusters_numbers
        )

        train_probas = self.get_features(train_loader, type="scores")
        test_probas = self.get_features(test_loader, type="scores")

        supervised_metrics_probas = metrics.get_supervised_metrics_probas(
            train_probas, test_probas, targets, clusters_numbers
        )

        all_metics = (
            supervised_metrics_features
            | unsupervised_metrics_features
            | supervised_metrics_probas
        )

        return all_metics

    def save_checkpoint(self, file_path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            file_path (str): Path to save the checkpoint.

        Returns:
            None
        """
        torch.save({"model_state_dict": self.models.state_dict()}, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """
        Load a checkpoint of the model.

        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        checkpoint = torch.load(file_path, weights_only=True)
        self.models.load_state_dict(checkpoint["model_state_dict"])
