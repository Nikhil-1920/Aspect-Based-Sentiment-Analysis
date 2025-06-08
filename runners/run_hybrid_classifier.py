import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the hybrid model from models/bert directory
from models.bert.HybridBERTClassifier import HybridABSA_Classifier

def main():
    os.makedirs("data/processed/bert_data", exist_ok=True)
    os.makedirs("outputs/hybrid", exist_ok=True)
    
    train_csv = "data/processed/bert_data/restaurant_train_features.csv"
    val_csv   = "data/processed/bert_data/restaurant_val_features.csv"
    output_dir = "outputs/hybrid/restaurant"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Hybrid Classifier
    print("Initializing Hybrid BERT Classifier...")
    hybrid_classifier = HybridABSA_Classifier(
        model_name="bert-base-uncased",
        num_labels=3,
        tfidf_dim=200,  
        use_attention=True
    )

    print(f"Training model with data from {train_csv}...")
    metrics = hybrid_classifier.train(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        batch_size=16,
        epochs=4,
        learning_rate=2e-5
    )

    print("Testing model on test data...")
    hybrid_classifier.load_model(os.path.join(output_dir, "best_model"))
    test_df = hybrid_classifier.predict_from_test_data(
        "data/processed/restaurant_test_features_unlabeled.csv",
        output_csv=os.path.join(output_dir, "predictions.csv")
    )
    
    print("Restaurant domain training and prediction complete!")
    
    # Now do the same for laptop domain
    train_csv = "data/processed/bert_data/laptop_train_features.csv"
    val_csv = "data/processed/bert_data/laptop_val_features.csv"
    output_dir = "outputs/hybrid/laptop"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Hybrid Classifier
    print("\nInitializing Hybrid BERT Classifier for laptop domain...")
    hybrid_classifier = HybridABSA_Classifier(
        model_name="bert-base-uncased",
        num_labels=3,
        tfidf_dim=200,  
        use_attention=True
    )

    print(f"Training model with data from {train_csv}...")
    metrics = hybrid_classifier.train(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        batch_size=16,
        epochs=4,
        learning_rate=2e-5
    )

    # Load the best model and predict
    print("Testing model on test data...")
    hybrid_classifier.load_model(os.path.join(output_dir, "best_model"))
    test_df = hybrid_classifier.predict_from_test_data(
        "data/processed/bert_data/laptop_test_features.csv",
        output_csv=os.path.join(output_dir, "predictions.csv")
    )
    
    print("Training and prediction complete for both domains!")

if __name__ == "__main__":
    main()
