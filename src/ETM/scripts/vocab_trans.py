import pickle
import argparse

parser = argparse.ArgumentParser(description="Convert a fairseq dictionary into an ETM vocab pickle.")
parser.add_argument("dict_file")
parser.add_argument("--stops", default="stops.txt")
parser.add_argument("--output", default="vocab.pkl")
args = parser.parse_args()

with open(args.dict_file, 'r') as f:
    line = f.readlines()
vocab = []
i = 0
while i <len(line):
    vocab.append(line[i].split(" ")[0])
    i += 1

with open(args.stops, 'r') as f:
    stops = f.read().split('\n')

vocab = [w for w in vocab if w not in stops]

with open(args.output, 'wb') as f:
    pickle.dump(vocab, f)
