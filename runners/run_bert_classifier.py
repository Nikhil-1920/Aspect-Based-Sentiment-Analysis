import os
import sys
import argparse
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the BERT model from the correct location
from models.bert.BERTModel import ABSA_BERT_Classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run BERT classifier for ABSA')
    parser.add_argument('--domains', nargs='+', default=['laptop', 'restaurant'], 
                        help='Domains to process (laptop, restaurant, or both)')
    parser.add_argument('--data_dir', default='data/processed/bert_data', 
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', default='outputs/bert', 
                        help='Directory to save model and results')
    parser.add_argument('--model', default='bert-base-uncased', 
                        help='BERT model to use')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=4, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                        help='Run mode: train, test, or both')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for domain in args.domains:
        domain_output = os.path.join(args.output_dir, domain)
        os.makedirs(domain_output, exist_ok=True)
        
        # Initialize BERT classifier
        logger.info(f"Training {domain} BERT model...")
        logger.info(f"Using model: {args.model}")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {domain_output}")
        
        bert_classifier = ABSA_BERT_Classifier(model_name=args.model)
        
        if args.mode in ['train', 'both']:
            logger.info(f"Training {domain} BERT model...")
            train_csv = os.path.join(args.data_dir, f"{domain}_train_features.csv")
            val_csv = os.path.join(args.data_dir, f"{domain}_val_features.csv")
            
            if not os.path.exists(train_csv):
                logger.error(f"Training file not found: {train_csv}")
                continue
                
            if not os.path.exists(val_csv):
                logger.error(f"Validation file not found: {val_csv}")
                continue
                
            logger.info(f"Training data: {train_csv}")
            logger.info(f"Validation data: {val_csv}")
            
            try:
                metrics = bert_classifier.train(
                    train_csv=train_csv,
                    val_csv=val_csv,
                    output_dir=domain_output,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.lr
                )
                
                logger.info(f"{domain} training complete. Metrics saved to {domain_output}")
            except Exception as e:
                logger.error(f"Error during training: {e}")
                continue
        
        if args.mode in ['test', 'both']:
            logger.info(f"Testing {domain} model...")
            
            # Load the best model
            model_path = os.path.join(domain_output, "best_model")
            
            if args.mode == 'test' or (args.mode == 'both' and not os.path.exists(model_path)):
                try:
                    bert_classifier.load_model(model_path)
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    continue
            
            test_csv = os.path.join(args.data_dir, f"{domain}_test_features_unlabeled.csv")
            output_csv = os.path.join(domain_output, f"{domain}_bert_predictions.csv")
            
            # Check if test file exists
            if not os.path.exists(test_csv):
                logger.error(f"Test file not found: {test_csv}")
                continue
                
            logger.info(f"Test data: {test_csv}")
            logger.info(f"Output file: {output_csv}")
            
            try:
                predictions = bert_classifier.predict_from_test_data(
                    test_csv=test_csv,
                    output_csv=output_csv
                )
                
                logger.info(f"{domain} predictions saved to {output_csv}")
            except Exception as e:
                logger.error(f"Error during testing: {e}")

if __name__ == "__main__":
    main()