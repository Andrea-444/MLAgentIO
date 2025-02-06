import pandas as pd

def eda(train: pd.DataFrame, test: pd.DataFrame):
    train["Text"].str.split().apply(len).plot.hist()
    train["Text"].str.split().apply(len).plot.box()
    test["Text"].str.split().apply(len).plot.hist()
    test["Text"].str.split().apply(len).plot.box()


def tokens_eda(train_embeding: pd.DataFrame, test_embeding: pd.DataFrame):
    plt.hist([len(tokens) for tokens in train_embeding])
    plt.hist([len(tokens) for tokens in test_embeding])
if __name__ == '__main__':
    pass