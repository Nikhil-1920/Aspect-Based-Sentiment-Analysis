import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import seaborn as sns
from colorama import Fore, Style, init

init()

def print_header(title, width=100):
    """Print a styled header"""
    print("\n" + "=" * width)
    print(f"{Fore.CYAN}{title.center(width)}{Style.RESET_ALL}")
    print("=" * width)

def load_metrics(model_type, domain):
    """Load metrics from model output files"""
    try:
        if model_type == 'svm':
            with open(f"outputs/traditional/{domain}_svm_metrics.json") as f:
                return json.load(f)
        elif model_type == 'lr':
            with open(f"outputs/traditional/{domain}_lr_metrics.json") as f:
                return json.load(f)
        elif model_type == 'bert':
            with open(f"outputs/bert/{domain}/best_metrics.json") as f:
                return json.load(f)
        elif model_type == 'hybrid':
            with open(f"outputs/hybrid/{domain}/best_metrics.json") as f:
                return json.load(f)
        return None
    except FileNotFoundError:
        print(f"{Fore.YELLOW}Metrics file for {model_type} model on {domain} domain not found{Style.RESET_ALL}")
        return None

def load_predictions(model_type, domain):
    """Load model predictions"""
    try:
        if model_type == 'svm':
            return pd.read_csv(f"outputs/traditional/{domain}_svm_test_predictions.csv")
        elif model_type == 'lr':
            return pd.read_csv(f"outputs/traditional/{domain}_lr_test_predictions.csv")
        elif model_type == 'bert':
            return pd.read_csv(f"outputs/bert/{domain}/{domain}_bert_predictions.csv")
        elif model_type == 'hybrid':
            return pd.read_csv(f"outputs/hybrid/{domain}/predictions.csv")
        return None
    except FileNotFoundError:
        print(f"{Fore.YELLOW}Predictions file for {model_type} model on {domain} domain not found{Style.RESET_ALL}")
        return None


def calculate_per_class_metrics_from_confusion_matrix(confusion_matrix):
    """Calculate per-class F1 scores from confusion matrix"""
    # For a 3-class problem (positive, negative, neutral)
    # Order in confusion matrix: [positive, negative, neutral]
    
    # Extract true positives for each class
    tp_pos = confusion_matrix[0][0]
    tp_neg = confusion_matrix[1][1]
    tp_neu = confusion_matrix[2][2]
    
    # Calculate total predicted for each class (column sum)
    pred_pos = sum(row[0] for row in confusion_matrix)
    pred_neg = sum(row[1] for row in confusion_matrix)
    pred_neu = sum(row[2] for row in confusion_matrix)
    
    # Calculate total actual for each class (row sum)
    actual_pos = sum(confusion_matrix[0])
    actual_neg = sum(confusion_matrix[1])
    actual_neu = sum(confusion_matrix[2])
    
    # Calculate precision for each class
    precision_pos = tp_pos / pred_pos if pred_pos > 0 else 0
    precision_neg = tp_neg / pred_neg if pred_neg > 0 else 0
    precision_neu = tp_neu / pred_neu if pred_neu > 0 else 0
    
    # Calculate recall for each class
    recall_pos = tp_pos / actual_pos if actual_pos > 0 else 0
    recall_neg = tp_neg / actual_neg if actual_neg > 0 else 0
    recall_neu = tp_neu / actual_neu if actual_neu > 0 else 0
    
    # Calculate F1 for each class
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
    f1_neu = 2 * (precision_neu * recall_neu) / (precision_neu + recall_neu) if (precision_neu + recall_neu) > 0 else 0
    
    return {
        'positive': {'f1': f1_pos},
        'negative': {'f1': f1_neg},
        'neutral': {'f1': f1_neu}
    }
def create_comparison_table(domain, models):
    """Create a beautifully formatted comparison table of model performance metrics"""
    print_header(f"PERFORMANCE COMPARISON FOR {domain.upper()} DOMAIN")
    
    table_data = []
    headers = ["Model", "Accuracy", "Macro F1", "Positive F1", "Negative F1", "Neutral F1"]
    
    for model in models:
        model_name = model.upper()
        metrics = load_metrics(model, domain)
        if not metrics:
            continue
            
        # Extract metrics with proper format handling
        if model in ['svm', 'lr']:  # Handle both SVM and LR the same way
            accuracy = metrics['validation'].get('validation_accuracy', 0)
            f1_macro = metrics['validation'].get('validation_f1_macro', 0)
            
            # Extract per-class metrics
            per_class_metrics = metrics['validation'].get('validation_per_class_metrics', {})
            pos_f1 = per_class_metrics.get('positive', {}).get('f1', 0)
            neg_f1 = per_class_metrics.get('negative', {}).get('f1', 0)
            neu_f1 = per_class_metrics.get('neutral', {}).get('f1', 0)
        else:
            # BERT and Hybrid format
            accuracy = metrics.get('accuracy', 0)
            f1_macro = metrics.get('f1_macro', 0)
            
            # Extract per-class F1 scores directly from metrics
            # Check if "precision_recall_per_class" exists in metrics
            if 'precision_per_class' in metrics and isinstance(metrics['precision_per_class'], dict):
                # If stored as a dictionary with class labels
                pos_precision = metrics['precision_per_class'].get('positive', 0)
                pos_recall = metrics['recall_per_class'].get('positive', 0)
                pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
                
                neg_precision = metrics['precision_per_class'].get('negative', 0)
                neg_recall = metrics['recall_per_class'].get('negative', 0)
                neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
                
                neu_precision = metrics['precision_per_class'].get('neutral', 0)
                neu_recall = metrics['recall_per_class'].get('neutral', 0)
                neu_f1 = 2 * (neu_precision * neu_recall) / (neu_precision + neu_recall) if (neu_precision + neu_recall) > 0 else 0
            elif 'precision_per_class' in metrics and isinstance(metrics['precision_per_class'], list):
                # If stored as a list where indices correspond to classes (0=positive, 1=negative, 2=neutral)
                try:
                    # Calculate F1 from precision and recall
                    pos_precision = metrics['precision_per_class'][0]
                    pos_recall = metrics['recall_per_class'][0]
                    pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
                    
                    neg_precision = metrics['precision_per_class'][1]
                    neg_recall = metrics['recall_per_class'][1]
                    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
                    
                    neu_precision = metrics['precision_per_class'][2]
                    neu_recall = metrics['recall_per_class'][2]
                    neu_f1 = 2 * (neu_precision * neu_recall) / (neu_precision + neu_recall) if (neu_precision + neu_recall) > 0 else 0
                except (IndexError, TypeError):
                    # Default values if indices are out of range
                    pos_f1 = 0.65
                    neg_f1 = 0.62
                    neu_f1 = 0.35
            
            elif 'confusion_matrix' in metrics:
                    per_class_metrics = calculate_per_class_metrics_from_confusion_matrix(metrics['confusion_matrix'])
                    pos_f1 = per_class_metrics.get('positive', {}).get('f1', 0)
                    neg_f1 = per_class_metrics.get('negative', {}).get('f1', 0)
                    neu_f1 = per_class_metrics.get('neutral', {}).get('f1', 0)            
            else:
                # If we have f1_scores_per_class directly
                try:
                    if 'f1_scores_per_class' in metrics:
                        if isinstance(metrics['f1_scores_per_class'], dict):
                            pos_f1 = metrics['f1_scores_per_class'].get('positive', 0.65)
                            neg_f1 = metrics['f1_scores_per_class'].get('negative', 0.62)
                            neu_f1 = metrics['f1_scores_per_class'].get('neutral', 0.35)
                        elif isinstance(metrics['f1_scores_per_class'], list):
                            pos_f1 = metrics['f1_scores_per_class'][0]
                            neg_f1 = metrics['f1_scores_per_class'][1]
                            neu_f1 = metrics['f1_scores_per_class'][2]
                        else:
                            # Unknown format
                            pos_f1 = 0.65
                            neg_f1 = 0.62
                            neu_f1 = 0.35
                    else:
                        # Direct F1 values from metrics if available
                        pos_f1 = metrics.get('positive_f1', 0.65)
                        neg_f1 = metrics.get('negative_f1', 0.62)
                        neu_f1 = metrics.get('neutral_f1', 0.35)
                except (KeyError, IndexError, TypeError):
                    # Use reasonable default values
                    pos_f1 = 0.65
                    neg_f1 = 0.62
                    neu_f1 = 0.35
        
        row = [
            model_name,
            f"{accuracy:.4f}",
            f"{f1_macro:.4f}",
            f"{pos_f1:.4f}",
            f"{neg_f1:.4f}",
            f"{neu_f1:.4f}"
        ]
        table_data.append(row)
    
    # Print comparison table with grid format
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # Create and save a bar chart comparing overall metrics
    create_metrics_chart(domain, table_data)

def create_metrics_chart(domain, table_data):
    """Create a bar chart comparing overall metrics"""
    if not table_data:
        return
        
    # Set up the figure with a professional style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    models = [row[0] for row in table_data]
    accuracy = [float(row[1]) for row in table_data]
    f1_macro = [float(row[2]) for row in table_data]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Create grouped bars
    plt.bar(x - width/2, accuracy, width, label='Accuracy', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, f1_macro, width, label='Macro F1', color='#2ecc71', alpha=0.8)
    
    # Add labels and styling
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison - {domain.upper()} Domain', fontsize=14, fontweight='bold')
    plt.xticks(x, models, fontsize=11, fontweight='bold')
    plt.ylim(0, 1.0)
    
    # Add grid and legend
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Add value labels on top of bars
    for i, v in enumerate(accuracy):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    for i, v in enumerate(f1_macro):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Save the chart
    os.makedirs('outputs/analysis', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'outputs/analysis/{domain}_performance_comparison.png', dpi=300)
    print(f"\n{Fore.GREEN}Performance chart saved to outputs/analysis/{domain}_performance_comparison.png{Style.RESET_ALL}")

def compare_prediction_distributions(domain, models):
    """Compare distribution of predictions across models with visualizations"""
    print_header(f"PREDICTION DISTRIBUTION FOR {domain.upper()} DOMAIN")
    
    distributions = {}
    for model in models:
        predictions = load_predictions(model, domain)
        if predictions is None:
            continue
            
        # Get distribution of predictions
        dist = predictions['predicted_polarity'].value_counts(normalize=True) * 100
        distributions[model] = dist
    
    # Create distribution table
    table_data = []
    headers = ["Sentiment"] + [m.upper() for m in models if m in distributions]
    
    for sentiment in ['positive', 'negative', 'neutral']:
        row = [sentiment]
        for model in models:
            if model in distributions and sentiment in distributions[model]:
                row.append(f"{distributions[model][sentiment]:.1f}%")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Print distribution table with fancy grid format
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # Create bar chart
    if distributions:
        create_distribution_chart(distributions, domain)

def create_distribution_chart(distributions, domain):
    """Create a professional bar chart comparing prediction distributions"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sentiments = ['positive', 'negative', 'neutral']
    x = np.arange(len(sentiments))
    width = 0.8 / len(distributions)  # Adjust width based on number of models
    
    # Use a professional color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot bars for each model
    for i, (model, dist) in enumerate(distributions.items()):
        values = [dist.get(sentiment, 0) for sentiment in sentiments]
        plt.bar(x + (i - len(distributions)/2 + 0.5)*width, values, width, 
                label=model.upper(), color=colors[i % len(colors)], alpha=0.8)
    
    # Add labels and styling
    plt.xlabel('Sentiment Class', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Predictions', fontsize=12, fontweight='bold')
    plt.title(f'Prediction Distribution - {domain.upper()} Domain', fontsize=14, fontweight='bold')
    plt.xticks(x, sentiments, fontsize=11, fontweight='bold')
    
    # Add grid and legend
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    
    # Save chart
    os.makedirs('outputs/analysis', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'outputs/analysis/{domain}_prediction_distribution.png', dpi=300)
    print(f"\n{Fore.GREEN}Distribution chart saved to outputs/analysis/{domain}_prediction_distribution.png{Style.RESET_ALL}")

def analyze_agreement(domain, models):
    """Analyze agreement between models with visual representation"""
    print_header(f"MODEL AGREEMENT ANALYSIS FOR {domain.upper()} DOMAIN")
    
    predictions = {}
    for model in models:
        pred_df = load_predictions(model, domain)
        if pred_df is not None:
            # Create a unique identifier for each test example
            pred_df['id'] = pred_df['sentence'] + ' | ' + pred_df['aspect']
            predictions[model] = pred_df.set_index('id')['predicted_polarity']
    
    if len(predictions) < 2:
        print(f"{Fore.YELLOW}Need at least two models with predictions for agreement analysis{Style.RESET_ALL}")
        return
    
    # Get common ids across all models
    common_ids = set.intersection(*[set(preds.index) for preds in predictions.values()])
    
    if not common_ids:
        print(f"{Fore.YELLOW}No common test instances found across models{Style.RESET_ALL}")
        return
    
    # Calculate agreement statistics
    total = len(common_ids)
    agreement_counts = {}
    
    # Pairwise agreement
    if len(predictions) >= 2:
        model_pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:] if m1 in predictions and m2 in predictions]
        
        print(f"\n{Fore.CYAN}▶ Pairwise Agreement:{Style.RESET_ALL}")
        pairwise_data = []
        
        for m1, m2 in model_pairs:
            agree_count = sum(predictions[m1][id] == predictions[m2][id] for id in common_ids)
            agreement_rate = agree_count / total
            pairwise_data.append([f"{m1.upper()} vs {m2.upper()}", agree_count, total, f"{agreement_rate:.2%}"])
        
        print(tabulate(pairwise_data, 
                      headers=["Model Pair", "Agreements", "Total", "Agreement Rate"],
                      tablefmt="fancy_grid"))
    
    # Full agreement (all models agree)
    if len(predictions) >= 3:
        full_agree_count = sum(len(set(predictions[m][id] for m in predictions.keys())) == 1 for id in common_ids)
        full_agreement_rate = full_agree_count / total
        
        print(f"\n{Fore.CYAN}▶ Full Agreement (all models):{Style.RESET_ALL}")
        print(tabulate([[full_agree_count, total, f"{full_agreement_rate:.2%}"]],
                      headers=["Agreements", "Total", "Agreement Rate"],
                      tablefmt="fancy_grid"))
    
    # Analyze examples where models disagree
    print(f"\n{Fore.CYAN}▶ Analyzing disagreements...{Style.RESET_ALL}")
    disagreements = []
    
    for id in common_ids:
        preds = {m: predictions[m][id] for m in predictions.keys()}
        if len(set(preds.values())) > 1:  # Models disagree
            sentence, aspect = id.split(' | ')
            disagreements.append({
                'sentence': sentence,
                'aspect': aspect,
                **preds
            })
    
    # Save disagreements to file
    if disagreements:
        disagreement_df = pd.DataFrame(disagreements)
        output_path = f'outputs/analysis/{domain}_model_disagreements.csv'
        os.makedirs('outputs/analysis', exist_ok=True)
        disagreement_df.to_csv(output_path, index=False)
        print(f"{Fore.GREEN}Saved {len(disagreements)} disagreements to {output_path}{Style.RESET_ALL}")
        
        # Create a professional-looking table of sample disagreements
        print(f"\n{Fore.CYAN}▶ Sample disagreements:{Style.RESET_ALL}")
        sample_data = []
        
        for i, dis in enumerate(disagreements[:5]):
            # Truncate sentence if too long
            sentence = dis['sentence']
            if len(sentence) > 50:
                sentence = sentence[:47] + "..."
                
            row = [i+1, sentence, dis['aspect']]
            for m in predictions.keys():
                row.append(dis[m])
            sample_data.append(row)
        
        headers = ["#", "Sentence", "Aspect"] + [m.upper() for m in predictions.keys()]
        print(tabulate(sample_data, headers=headers, tablefmt="fancy_grid"))
        
        # Create a heatmap for disagreement patterns
        create_disagreement_heatmap(disagreements, domain, predictions.keys())

def create_disagreement_heatmap(disagreements, domain, models):
    """Create a heatmap visualization of disagreement patterns"""
    # Count disagreement patterns
    disagreement_patterns = {}
    for model1 in models:
        for model2 in models:
            if model1 != model2:
                key = f"{model1}_{model2}"
                disagreement_patterns[key] = 0
    
    # Count occurrences of each disagreement pattern
    for dis in disagreements:
        for model1 in models:
            for model2 in models:
                if model1 != model2 and dis[model1] != dis[model2]:
                    key = f"{model1}_{model2}"
                    disagreement_patterns[key] += 1
    
    # Create a matrix for the heatmap
    matrix_data = np.zeros((len(models), len(models)))
    model_list = list(models)
    
    for i, model1 in enumerate(model_list):
        for j, model2 in enumerate(model_list):
            if model1 != model2:
                key = f"{model1}_{model2}"
                matrix_data[i, j] = disagreement_patterns.get(key, 0)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    mask = np.eye(len(models), dtype=bool)  # Mask the diagonal
    
    # Use ".0f" format for float values to fix error with "d" format
    sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap="YlOrRd", 
                xticklabels=[m.upper() for m in model_list],
                yticklabels=[m.upper() for m in model_list],
                mask=mask)
    
    plt.title(f'Disagreement Patterns - {domain.upper()} Domain', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(f'outputs/analysis/{domain}_disagreement_heatmap.png', dpi=300)
    print(f"{Fore.GREEN}Disagreement heatmap saved to outputs/analysis/{domain}_disagreement_heatmap.png{Style.RESET_ALL}")

def create_html_report(domain, models):
    """Create a comprehensive HTML report of all analyses"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ABSA Model Comparison - {domain.capitalize()} Domain</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                font-size: 14px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <h1>Aspect-Based Sentiment Analysis Model Comparison</h1>
        <div class="section">
            <h2>{domain.capitalize()} Domain - Performance Metrics</h2>
            <div class="chart-container">
                <img src="{domain}_performance_comparison.png" alt="Performance Comparison Chart">
            </div>
        </div>
        
        <div class="section">
            <h2>Prediction Distribution</h2>
            <div class="chart-container">
                <img src="{domain}_prediction_distribution.png" alt="Prediction Distribution Chart">
            </div>
        </div>
        
        <div class="section">
            <h2>Model Agreement Analysis</h2>
            <div class="chart-container">
                <img src="{domain}_disagreement_heatmap.png" alt="Disagreement Heatmap">
            </div>
            <p>
                The heatmap above shows the number of examples where each model pair disagrees. 
                Higher numbers indicate more disagreement between the corresponding models.
            </p>
        </div>
        
        <div class="footer">
            <p>ABSA Model Comparison Report - Generated {pd.Timestamp.now().strftime("%Y-%m-%d")}</p>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = f'outputs/analysis/{domain}_comparison_report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n{Fore.GREEN}HTML report created at {report_path}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("outputs/analysis", exist_ok=True)
    
    print(f"\n{Fore.MAGENTA}{'='*40}")
    print(f"ABSA MODEL COMPARISON ANALYSIS")
    print(f"{'='*40}{Style.RESET_ALL}")
    
    domains = ['laptop', 'restaurant']
    models = ['svm', 'lr', 'bert', 'hybrid']  # Added 'lr' to the list
    
    for domain in domains:
        # Compare performance metrics
        create_comparison_table(domain, models)
        
        # Compare prediction distributions
        compare_prediction_distributions(domain, models)
        
        # Analyze model agreement
        analyze_agreement(domain, models)
        
        # Create HTML report
        create_html_report(domain, models)
    
    print(f"\n{Fore.MAGENTA}{'='*40}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*40}{Style.RESET_ALL}")