import logging
from src.dataset import load_dataset
from src.model import TicketTriageModel

def main():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    ## Prepare datasets
    logging.info("Prepare datasets...")
    dataset, queue_labels, id2label, label2id, class_weight_dict = load_dataset()
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    ## Prepare base model
    model = TicketTriageModel(id2label=id2label, model_name="fastino/gliner2-large-v1")

    zshot_dataset = model.get_predictions_from_dataset(test_dataset, batch_size=64)
    zshot_dataset.to_parquet(f"output/{model.model_name.replace('/',"_")}_preds.parquet")
    
    # From a saved predictions file
    # import datasets as ds
    # finetuned_dataset = ds.Dataset.from_parquet(f"output/{model.model_name.replace('/',"_")}_preds.parquet")

    logging.info(zshot_dataset.to_pandas().head())

    zshot_accuracy = zshot_dataset.filter(lambda x: x["pred_labels"] == x["labels_str"]).num_rows * 100 / zshot_dataset.num_rows
    logging.info(zshot_accuracy)


if __name__ == "__main__":
    main()