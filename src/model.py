from typing import Any
import logging
from gliner2 import GLiNER2
import torch
import datasets as ds

class TicketTriageModel:
    def __init__(
            self,
            id2label: dict[int, Any],
            model_name: str = "fastino/gliner2-base-v1",
            threshold: float = 0.65       
            ):
        logging.basicConfig(
           level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.model_name = model_name
        self.extractor = GLiNER2.from_pretrained(self.model_name)
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        torch.cuda.empty_cache() if torch.cuda.is_available() else torch.xpu.empty_cache() if torch.xpu.is_available() else "pass"
        logging.info(f"Device: {self.device}")
        self.id2label = id2label
        self.labels = [l.replace(" ","_").lower() for l in list(id2label.values())]
        logging.info(f"Labels: {self.labels}")
        self.schema= {"ticket_type": self.labels}
        ## Optionally one may use the descriptions
        self.schema_with_description = { "ticket_type": {
                "technical_support": "Technical issues and support requests",
                "customer_service": "customer inquiries and service requests",
                "billing_and_payments": "Billing issues and payment processing",
                "product_support": "Support for product-related issues",
                "it_support": "Internal IT support and infrastructure issues",
                "returns_and_exchanges": "Product returns and exchanges",
                "sales_and_pre_sales": "Sales inquiries and pre-sales questions",
                "human_resources": "Employee inquiries and HR-related issues",
                "service_outages_and_maintenance": "Service interruptions and maintenance",
                "general_inquiry": "General inquiries and information requests"
            }
        } 
        logging.info(f"Schema: {self.schema}")

    def get_predictions_from_dataset(self, dataset: ds.Dataset, batch_size: int = 32) -> ds.Dataset:
        """
        Run batch inference on a Hugging Face Dataset and add predictions as a column.
        """
        def ids2labels(batch):
            return {"labels_str" : [self.id2label[_id].replace(" ","_").lower() for _id in batch["labels"]]} 

        def predict(batch):
            outputs = self.extractor.batch_classify_text(
                texts=batch["text"],
                tasks=self.schema,
                batch_size=batch_size,
                threshold=self.threshold,
                format_results=True
            )
            logging.info({"pred_labels": [list(out.values())[0] for out in outputs]})
            return {"pred_labels": [list(out.values())[0] for out in outputs]}
        
        dataset = dataset.map(predict, batched=True, batch_size=batch_size)
        dataset = dataset.map(ids2labels, batched=True, batch_size=batch_size)
        return dataset