# -*- coding: utf-8 -*-
# Attention! The model is only available for Chinese names
import os
import torch
import pickle
import datetime
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from joblib import dump,load
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class NameSexNew:
    def __init__(self, c2=10, n_jobs=-1, n_estimators = 500,
                 w2v_filename=('namesex/model/w2v_dictvec_sg_s100i20.pickle')):

        self.w2v_dictvec = None
        self.vectorizer = None
        self.w2v_way_rf = 'diverge'
        self.x_array = None
        self.lg = LogisticRegression(C=c2)
        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        self.input_size = None
        self.lstm_model = None
        self.train_loader = None
        # define w2v related
        with open(w2v_filename, "rb") as f:
            self.w2v_dictvec = pickle.load(f)
        firstchar = next(iter(self.w2v_dictvec.keys()))
        self.w2v_vecsize = len(self.w2v_dictvec[firstchar])

    # Define LSTM Class
    class lstm(nn.Module):
        def __init__(self, input_size, rnn_hidden_size, hidden_size, output_size=2):
            super().__init__()
            self.drop = nn.Dropout(p=0.1)
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=rnn_hidden_size,
                               batch_first=True)
            self.linear1 = nn.Linear(rnn_hidden_size, hidden_size)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size, output_size)
            self.output_size = output_size
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, name, length):
            now = name
            now = self.drop(now)
            input_packed = nn.utils.rnn.pack_padded_sequence(now, length, batch_first=True, enforce_sorted=False)
            _, (ht, _) = self.rnn(input_packed, None)
            out = self.linear1(ht)
            out = self.activation(out)
            out = self.linear2(out)

            out = out.view(-1, self.output_size)
            out = self.softmax(out)
            return out

    ## Preprocess data and preparation ##
    # Get vector library
    def get_vector_lib(self, namevec, min_gname=2):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1), min_df=min_gname)
        self.vectorizer.fit(namevec)
    # Extract_w2v for random forest
    def extract_w2v(self, namearray):
        num_names = len(namearray)
        xw2v1 = np.zeros((num_names, self.w2v_vecsize))
        xw2v2 = np.zeros((num_names, self.w2v_vecsize))

        for i, aname in enumerate(namearray):
            vectors = [self.w2v_dictvec[achar] for achar in aname if achar in self.w2v_dictvec]
            if vectors:
                vectors = [vec / np.linalg.norm(vec) for vec in vectors]
                vec_mean = np.mean(vectors, axis=0)
                vec_diverge1 = np.maximum.reduce(vectors)
                vec_diverge0 = np.minimum.reduce(vectors)
                xw2v1[i] = vec_mean
                xw2v2[i] = vec_diverge1 + vec_diverge0

        return xw2v1, xw2v2
    ## Name to Vector ##
    # Logistic Regression Name to Vector
    def gen_feat(self, namevec):
        x_array = self.vectorizer.transform(namevec).toarray()
        self.x_array = x_array
        return x_array
    # Random Forest Name to Vector
    def gen_feat_rf(self, namevec):
        x_array = self.gen_feat(namevec)
        w2v1, w2v2 = self.extract_w2v(namevec)

        if self.w2v_way_rf == "mean":
            x_array_rf = np.concatenate((x_array, w2v1), axis=1)
        else:
            x_array_rf = np.concatenate((x_array, w2v2), axis=1)
        return x_array_rf

    # LSTM Name to Vector Word Count + w2v
    def name_to_vec_w2v(self, name):
        vecs = []
        count_vec = self.vectorizer.transform([name]).toarray()
        w2v_vec = self.w2v_dictvec.get(name, None)
        if w2v_vec is None:
            w2v_vec = np.zeros(100)
        combined_vec = np.concatenate([count_vec[0], w2v_vec])
        vecs.append(torch.tensor(combined_vec))
        return torch.stack(vecs)

    # LSTM value to tensor
    def value_to_tensor(self, namevec):
        name_tensors_list = [self.name_to_vec_w2v(name) for name in namevec]
        lengths = torch.tensor([len(t) for t in name_tensors_list])
        max_len = max(lengths)
        name_tensors = torch.zeros(len(namevec), max_len, self.input_size)  # Shape: [batch, max_len, embedding_dim]

        for i, tensor in enumerate(name_tensors_list):
            name_tensors[i, :tensor.shape[0], :] = tensor

        return name_tensors, lengths

    # LSTM Setting up
    def train_lstm(self, epochs, input_size, lr = 0.001, rnn_hidden_size = 128, hidden_size = 50):
        self.lstm_model = self.lstm(input_size,rnn_hidden_size, hidden_size)
        optimizer = optim.RMSprop(self.lstm_model.parameters(), lr=lr)
        loss_func = nn.NLLLoss()

        for epoch in range(epochs):
            total_loss = 0
            for names, labels in self.train_loader:
                self.lstm_model.train()
                optimizer.zero_grad()
                name_tensors, name_lengths = self.value_to_tensor(names)
                labels = torch.tensor(labels, dtype=torch.long)
                outputs = self.lstm_model(name_tensors, name_lengths).view(-1, 2)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            acc, true_labels, predicted_labels = self.test_model()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # LSTM Training process tester
    def test_model(self):
        self.lstm_model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for names, labels in self.train_loader:
                name_tensors, name_lengths = self.value_to_tensor(names)
                labels = torch.tensor(labels, dtype=torch.long)
                outputs = self.lstm_model(name_tensors, name_lengths).view(-1, 2)
                predicted = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())
        accuracy = sum([1 if true == pred else 0 for true, pred in zip(all_labels, all_predictions)]) / len(all_labels)
        return accuracy, all_labels, all_predictions

    ## Model Training ##
    def train(self, namevec, sexvec):
        # Get the start time
        start_time = time.time()

        # Train Test Setting
        x_train_lg = self.gen_feat(namevec)
        x_train_rf = self.gen_feat_rf(namevec)
        x_train_lstm = list(zip(namevec, sexvec))
        self.train_loader = DataLoader(x_train_lstm, batch_size= 128, shuffle=True)
        self.input_size = self.name_to_vec_w2v(x_train_lstm[0][0]).shape[1]

        y_train = sexvec
        # Logistic Regression
        print('Training Logistic Regression')
        self.lg.fit(x_train_lg,y_train)
        print('Training Completed')

        # Random Forest
        print('Training Random Forest')
        self.rf.fit(x_train_rf,y_train)
        print('Training Completed')

        # LSTM
        print('Training LSTM')
        self.train_lstm(epochs=20,input_size=self.input_size,lr=0.005)
        print('Training Completed')

        elapsed_time = time.time() - start_time
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        with open('namesex/report/training_time_log.txt', 'a') as f:
            f.write(f"{current_time}, {elapsed_time:.2f} seconds\n")
    def run_train(self,):
        df = pd.read_csv('namesex/data/namesex_data_v2.csv')
        df = df[['gname', 'sex']]
        namevec = df['gname']
        sexvec = df['sex']
        self.get_vector_lib(namevec)
        print("Training models...")
        self.train(namevec,sexvec)
        print("All models have been trained.")
    ## Prediction ##
    def predict(self, model, namevec, predprob = False):
        if model == 'rf':
            x2_test = self.gen_feat_rf(namevec)
            ypred_prob_rf = self.rf.predict_proba(x2_test)
            if predprob == True:
                return ypred_prob_rf[:,1]
            else:
                return np.asarray(ypred_prob_rf[:,1] >=0.5, "int")
        elif model == 'lg':
            x_test = self.gen_feat(namevec)
            ypred_prob = self.lg.predict_proba(x_test)
            if predprob == True:
                return ypred_prob[:,1]
            else:
                return np.asarray(ypred_prob[:,1] >=0.5, "int")
        else:
            self.input_size = self.name_to_vec_w2v(namevec[0]).shape[1]
            test_name_tensors, test_name_lengths = self.value_to_tensor(namevec)
            with torch.no_grad():
                outputs = self.lstm_model(test_name_tensors, test_name_lengths)
            if predprob == True:
                probabilities = torch.exp(outputs)
                return probabilities[0]
            else:
                predicted_class = torch.argmax(outputs, dim=1).numpy()
                return predicted_class
    ## Load & Save ##
    def load_model(self,filename = 'namesex/model/'):
        self.rf = load(filename + 'random_forest_model.joblib')
        self.lg = load(filename + 'logistic_regression_model.joblib')
        self.lstm_model = torch.load(filename + 'lstm.pth')
    def save_model(self,):
        dump(self.lg, 'namesex/model/logistic_regression_model.joblib')
        dump(self.rf, 'namesex/model/random_forest_model.joblib')
        torch.save(self.lstm_model, 'namesex/model/lstm.pth')




    ## k-fold validation
    from sklearn.metrics import log_loss
    def k_fold_validation(self, namevec, sexvec, n_splits=10):

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        model_reports = {'lg': [], 'rf': [], 'lstm': []}
        model_log_losses = {'lg': [], 'rf': [], 'lstm': []}  # To store log loss for each model

        fold_number = 1  # Initialize fold number

        for train_idx, test_idx in kfold.split(namevec):
            print(f"\n{'-' * 50}")
            print(f"Training progress: {fold_number}/{n_splits}")  # Display the progress
            print(f"{'-' * 50}\n")

            train_namevec, test_namevec = namevec[train_idx], namevec[test_idx]
            train_sexvec, test_sexvec = sexvec[train_idx], sexvec[test_idx]

            self.get_vector_lib(train_namevec)
            self.train(train_namevec, train_sexvec)

            for model_name in model_reports.keys():
                y_pred = self.predict(model_name, test_namevec)
                report = classification_report(test_sexvec, y_pred, output_dict=True)
                model_reports[model_name].append(report)

            fold_number += 1  # Increment the fold number

        output_reports = {}
        for model_name, reports in model_reports.items():
            performance_table = {'Metric': [], 'Performance': [], 'Performance Std. Dev.': []}

            # For accuracy, precision, recall, F1
            for key in ['accuracy', 'macro avg']:
                if key == 'accuracy':
                    avg_value = np.mean([r[key] for r in reports])
                    std_value = np.std([r[key] for r in reports])
                    performance_table['Metric'].append('Accuracy')
                    performance_table['Performance'].append(avg_value)
                    performance_table['Performance Std. Dev.'].append(std_value)
                else:
                    avg_value = np.mean([r[key]['f1-score'] for r in reports])
                    std_value = np.std([r[key]['f1-score'] for r in reports])
                    performance_table['Metric'].append('F1')
                    performance_table['Performance'].append(avg_value)
                    performance_table['Performance Std. Dev.'].append(std_value)

                    avg_value = np.mean([r[key]['precision'] for r in reports])
                    std_value = np.std([r[key]['precision'] for r in reports])
                    performance_table['Metric'].append('Precision')
                    performance_table['Performance'].append(avg_value)
                    performance_table['Performance Std. Dev.'].append(std_value)

                    avg_value = np.mean([r[key]['recall'] for r in reports])
                    std_value = np.std([r[key]['recall'] for r in reports])
                    performance_table['Metric'].append('Recall')
                    performance_table['Performance'].append(avg_value)
                    performance_table['Performance Std. Dev.'].append(std_value)

            output_reports[model_name] = performance_table

        return output_reports


def main():
    df = pd.read_csv('namesex/data/namesex_data_v2.csv')
    df = df[['gname', 'sex']]
    ns = NameSexNew()
    ns.get_vector_lib(df['gname'])

    mode = input("Choose a mode: 1 - predict, 2 - explore, 3 - train, 4 - test:, 5 - k_fold_validation ")

    if mode not in ["3", "train"]:
        try:
            ns.load_model()
        except:
            print("Model not found! Please enter 3 or train to train the model first.")
            return

    if mode == "1" or mode == "predict":
        model_choice = input("Which model do you want to use? (lg: Logistic Regression, rf: Random Forest, lstm): ")

        if model_choice in ['lg', 'rf', 'lstm']:
            ns.load_model()
            name_to_predict = input("Enter a name to predict its gender (or type 'exit' to stop): ")
            if name_to_predict.lower() == 'exit':
                pass
            else:
                prediction = ns.predict(model_choice, [name_to_predict])
                if prediction[0] == 1:
                    print(f"The predicted gender for {name_to_predict} is Male.")
                else:
                    print(f"The predicted gender for {name_to_predict} is Female.")
                want_prob = input("Do you want to know the probability? (y/n): ")
                if want_prob.lower() == 'y':
                    if model_choice == 'lstm':
                        prob = ns.predict(model_choice, [name_to_predict], predprob=True)
                        print(f"Probability for {name_to_predict}: Female = {prob[0]*100:.4f}%, Male = {prob[1]*100:.4f}%")
                    else:
                        prob_array = ns.predict(model_choice, [name_to_predict], predprob=True)
                        prob_value = prob_array[0]  # Extract the first value from the numpy array
                        predicted_gender = "Male" if prediction[0] >= 0.5 else "Female"
                        print(f"The predicted gender for {name_to_predict} is {predicted_gender} with a probability of {prob_value * 100:.4f}%.")
                else:
                    print("Thank you!")
        else:
            print("Invalid model choice!")

    elif mode == "2" or mode == "explore":

        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        ns.train(train_df['gname'], train_df['sex'])
        for model_name in ['lg', 'rf', 'lstm']:
            predicted = ns.predict(model_name, test_df['gname'])
            accuracy = accuracy_score(test_df['sex'], predicted)
            print(f"\n{'=' * 20}")
            print(f"Model: {model_name.upper()}")
            print(f"{'=' * 20}\n")
            print(f"Accuracy: {accuracy * 100:.4f}%")
            print(classification_report(test_df['sex'], predicted))

    elif mode == "3" or mode == "train":

        ns.run_train()
        ns.save_model()

    elif mode == "4" or mode == "test":

        test_df = pd.read_csv('namesex/data/testdata.csv')
        test_df = test_df[['name', 'sex']]
        ns.load_model()
        # test result display
        for model_name in ['lg', 'rf', 'lstm']:
            predicted = ns.predict(model_name, test_df['name'])
            print(f"\n{'=' * 20}")
            print(f"Model: {model_name.upper()}")
            print(f"{'=' * 20}\n")
            names = test_df['name'][0:5].to_list()
            actual_sex = test_df['sex'][0:5].to_list()
            print("The first 5 given names are:")
            for i, (name, sex, pred) in enumerate(zip(names, actual_sex, predicted)):
                print(f"Name: {name:<15} Actual Sex: {sex:<7} Predicted Sex: {pred}")

        # classification report
        for model_name in ['lg', 'rf', 'lstm']:
            predicted = ns.predict(model_name, test_df['name'])
            accuracy = accuracy_score(test_df['sex'], predicted)
            print(f"\n{'=' * 20}")
            print(f"Model: {model_name.upper()}")
            print(f"{'=' * 20}\n")
            print(f"Accuracy: {accuracy * 100:.4f}%")
            print(classification_report(test_df['sex'], predicted))

    elif mode == "5" or mode == "validation":
        if os.path.exists('namesex/report/avg_reports.pkl'):
            with open('namesex/report/avg_reports.pkl', 'rb') as f:
                avg_reports = pickle.load(f)
        else:
            avg_reports = ns.k_fold_validation(df['gname'].to_numpy(), df['sex'].to_numpy())
            with open('namesex/report/avg_reports.pkl', 'wb') as f:
                pickle.dump(avg_reports, f)

        output_lines=[]
        for model_name, report in avg_reports.items():
            header = f"Model: {model_name.upper()}"
            print('\n' + '=' * len(header))
            print(header)
            print('=' * len(header) + '\n')

            # Using formatted string to control width and align right for better presentation
            table_header = "{:<15} {:<15} {:<25}".format("Metric", "Performance", "Performance Std. Dev.")
            print(table_header)
            print('-' * 55)

            # Add table header to the list
            output_lines.append(table_header)

            metrics = report['Metric']
            performances = report['Performance']
            std_devs = report['Performance Std. Dev.']

            for metric, perf, std_dev in zip(metrics, performances, std_devs):
                line = "{:<15} {:<15.4f} {:<25.6f}".format(metric, perf, std_dev)
                print(line)

                # Add each line to the list
                output_lines.append(line)

            # Save to a file
        with open('namesex/report/k_fold_report.txt', 'w') as f:
            for line in output_lines:
                f.write(line + '\n')

    else:

        print("Invalid mode selected!")

if __name__ == "__main__":
    main()
