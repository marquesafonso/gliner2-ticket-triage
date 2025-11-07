import datasets as ds
import kagglehub, logging


def load_dataset():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    dataset = "tobiasbueck/multilingual-customer-support-tickets"
    subset = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    # Download latest version
    kagglehub.dataset_download(dataset)

    # Load a DataFrame with a specific version of a CSV
    df: ds.Dataset = kagglehub.dataset_load(
        adapter = kagglehub.KaggleDatasetAdapter.HUGGING_FACE,
        handle = dataset,
        path = subset
    )
    seed = 10
    # df = df.to_iterable_dataset()

    df_en = df.filter(lambda x: x["language"] == "en")
    df_en = df_en.select_columns(["subject", "body", "queue"])
    df_en = df_en.map(lambda x: {
        "subject": x.get("subject", "") or "",
        "body": x.get("body", "") or "",
        "queue": x.get("queue")
    })
    df_en = df_en.map(lambda x: {
        "ticket": x.get("subject") + " " + x.get("body")
        })

    df_en = df_en.class_encode_column("queue")
    df_en = df_en.select_columns(["ticket", "queue"]).rename_columns({"ticket": "text", "queue": "labels"})
    logging.info(df_en.to_pandas())

    ## Creating label mappings
    id2label = {i: label for i, label in enumerate(df_en.features["labels"].names)}
    label2id = {label: i for i, label in enumerate(df_en.features["labels"].names)}
    queue_labels = list(label2id.keys())

    ## Splitting dataset into train and test sets
    train, test = df_en.train_test_split(test_size=0.1, stratify_by_column="labels", seed=seed).values()
    train = train.shuffle(seed=seed)

    ## Verifying distribution of class labels in train and validation datasets
    labels = sorted(train.to_pandas()["labels"].unique())
    class_weight_dict = {}
    for l in labels:
        class_weight_dict.update({l: train.to_pandas()["labels"].apply(lambda x: x == l).sum() / train.to_pandas()["labels"].count()})
        logging.info(f"[Train] Label {l}: {train.to_pandas()["labels"].apply(lambda x: x == l).sum()} occurrences")

    dataset_dict = ds.DatasetDict({
        "train": train,
        "test": test
    })

    return dataset_dict, queue_labels, id2label, label2id, class_weight_dict