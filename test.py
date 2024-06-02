import boto3
import pickle

client = boto3.client('s3')
path = "s3://emotion-text"

with open(path + 'model.pkl', 'rb') as file:
    model = pickle.load(file)

print(model)