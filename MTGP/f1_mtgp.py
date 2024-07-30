import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import bernoulli
import itertools
from tqdm import tqdm
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import ast



def calculate_pns_bounds(P_yx, P_y_x_prime, P_y_prime_x_prime, P_y, P_y_and_x, P_y_prime_and_x, P_y_and_x_prime, P_y_prime_and_x_prime):
    sigma = 1
    W = ((P_yx) - 2 * (P_y_x_prime) - (P_y_prime_and_x_prime))
    L = max(0, P_yx - P_y_x_prime, P_y - P_y_x_prime, P_yx - P_y)
    U = min(P_yx, P_y_prime_x_prime, P_y_and_x + P_y_prime_and_x_prime, P_yx - P_y_x_prime + P_y_and_x_prime + P_y_prime_and_x)
    lower_bound = (W + sigma * U)
    upper_bound = (W + sigma * L)
    return lower_bound, upper_bound

def calculate_predicted_benefit(preds):
    lower_bound, upper_bound = calculate_pns_bounds(*preds)
    return (lower_bound + upper_bound) / 2

class MTGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        num_tasks = train_y.shape[-1]
        super(MTGPModel, self).__init__(train_x, train_y, likelihood)
        

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        self.data_covar_module = RBFKernel()
        self.task_covar_module = IndexKernel(num_tasks=num_tasks, rank=1)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.data_covar_module, num_tasks=num_tasks, rank=1
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

def train_model(model, likelihood, train_x, train_y, num_epochs):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, likelihood, data_loader, benefit_mean):
    model.eval()
    likelihood.eval()
    total_loss = 0.0
    count = 0
    correct = 0
    total = 0
    mse_loss = torch.nn.MSELoss()

    all_predicted_group_labels = []
    all_actual_group_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            preds = model(inputs)
            mean_preds = preds.mean
            loss = mse_loss(mean_preds, labels)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Calculate group labels based on predicted p1 to p8 using calculate_predicted_benefit
            predicted_benefits = [calculate_predicted_benefit(pred.numpy()) for pred in mean_preds]
            predicted_group_labels = [1 if benefit >= benefit_mean else 0 for benefit in predicted_benefits]

            # Compare predicted group labels with actual group labels
            actual_benefits = [calculate_predicted_benefit(label.numpy()) for label in labels]
            actual_group_labels = [1 if benefit >= benefit_mean else 0 for benefit in actual_benefits]

            all_predicted_group_labels.extend(predicted_group_labels)
            all_actual_group_labels.extend(actual_group_labels)

            correct += (np.array(predicted_group_labels) == np.array(actual_group_labels)).sum()
            total += len(actual_group_labels)

    accuracy = correct / total
    f1 = f1_score(all_actual_group_labels, all_predicted_group_labels)
    return total_loss / count, accuracy, f1

def main_model_training():
    df = pd.read_csv('MT_1_8.csv')
    features = [ast.literal_eval(observed)[:15] for observed in df['subpopulation_values']]
    labels = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']].values
    benefit_mean = df['benefit'].mean()
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    train_x = torch.cat([batch[0] for batch in train_loader], dim=0)
    train_y = torch.cat([batch[1] for batch in train_loader], dim=0)

    likelihood = MultitaskGaussianLikelihood(num_tasks=8)
    model = MTGPModel(train_x, train_y, likelihood)
    
    train_model(model, likelihood, train_x, train_y, num_epochs=600)
    val_loss, val_accuracy, val_f1 = evaluate_model(model, likelihood, val_loader, benefit_mean)
    print(f'Validation MSE Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')

    test_df = pd.read_csv('test_data_mt.csv')
    test_df['subpopulation_values'] = test_df['subpopulation_values'].apply(ast.literal_eval)
    test_features = scaler.transform(test_df['subpopulation_values'].tolist())
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_df[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']].values, dtype=torch.float32)
    test_group_labels = torch.tensor(test_df['group_label'].values, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(test_features, test_labels, test_group_labels)
    test_loader = DataLoader(test_dataset, batch_size=64)
    test_loss, test_accuracy, test_f1 = evaluate_model(model, likelihood, test_loader, benefit_mean)
    print(f'Test MSE Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}')

if __name__ == "__main__":
    main_data_generation()
    main_test_data_generation()
    main_model_training()
