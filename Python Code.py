import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CustomerBehaviorAnalyzer:
    def __init__(self):
        self.customers_df = None
        self.transactions_df = None
        self.products_df = None
        self.sessions_df = None
        
    def generate_sample_data(self, n_customers=1000, n_products=50, n_transactions=5000):
        """Generate sample customer behavior data"""
        
        # Generate customers data
        np.random.seed(42)
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.randint(18, 70, n_customers),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers, p=[0.5, 0.35, 0.15]),
            'registration_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) 
                                for _ in range(n_customers)],
            'customer_segment': np.random.choice(['Premium', 'Regular', 'Budget'], n_customers, p=[0.2, 0.5, 0.3])
        }
        self.customers_df = pd.DataFrame(customer_data)
        
        # Generate products data
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
        product_data = {
            'product_id': range(1, n_products + 1),
            'category': np.random.choice(categories, n_products),
            'price': np.random.uniform(10, 500, n_products).round(2),
            'brand': [f'Brand_{i}' for i in np.random.randint(1, 21, n_products)]
        }
        self.products_df = pd.DataFrame(product_data)
        
        # Generate transactions data
        transaction_data = {
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
            'product_id': np.random.randint(1, n_products + 1, n_transactions),
            'quantity': np.random.randint(1, 5, n_transactions),
            'transaction_date': [datetime.now() - timedelta(days=np.random.randint(1, 180)) 
                               for _ in range(n_transactions)],
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], 
                                            n_transactions, p=[0.4, 0.3, 0.2, 0.1])
        }
        self.transactions_df = pd.DataFrame(transaction_data)
        
        # Add transaction amount
        self.transactions_df = self.transactions_df.merge(self.products_df[['product_id', 'price']], on='product_id')
        self.transactions_df['amount'] = self.transactions_df['quantity'] * self.transactions_df['price']
        
        # Generate web session data
        n_sessions = n_transactions * 2
        session_data = {
            'session_id': range(1, n_sessions + 1),
            'customer_id': np.random.randint(1, n_customers + 1, n_sessions),
            'session_date': [datetime.now() - timedelta(days=np.random.randint(1, 180)) 
                           for _ in range(n_sessions)],
            'pages_viewed': np.random.randint(1, 20, n_sessions),
            'session_duration': np.random.randint(30, 3600, n_sessions),  # seconds
            'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], 
                                          n_sessions, p=[0.4, 0.5, 0.1])
        }
        self.sessions_df = pd.DataFrame(session_data)
        
        # Verify data integrity
        print("Sample data generated successfully!")
        print(f"Customers: {len(self.customers_df)}")
        print(f"Products: {len(self.products_df)}")
        print(f"Transactions: {len(self.transactions_df)}")
        print(f"Sessions: {len(self.sessions_df)}")
        
        # Check for any data quality issues
        if self.transactions_df['customer_id'].max() > len(self.customers_df):
            print("Warning: Some transactions reference non-existent customers")
        if self.transactions_df['product_id'].max() > len(self.products_df):
            print("Warning: Some transactions reference non-existent products")
    
    def calculate_rfm_metrics(self):
        """Calculate RFM (Recency, Frequency, Monetary) metrics for customers"""
        
        # Calculate metrics
        current_date = self.transactions_df['transaction_date'].max()
        
        rfm = self.transactions_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Create RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
        
        # Combine RFM scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Segment customers based on RFM
        def segment_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['rfm_score'] in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '125', '124']:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        
        return rfm
    
    def analyze_customer_segments(self):
        """Analyze customer segments and purchasing patterns"""
        
        # Customer demographics analysis
        demo_analysis = self.customers_df.groupby('customer_segment').agg({
            'age': 'mean',
            'customer_id': 'count'
        }).round(2)
        demo_analysis.columns = ['avg_age', 'count']
        
        # Purchase behavior by segment
        purchase_behavior = self.transactions_df.merge(
            self.customers_df[['customer_id', 'customer_segment']], 
            on='customer_id'
        ).groupby('customer_segment').agg({
            'amount': ['sum', 'mean', 'count'],
            'quantity': 'sum'
        }).round(2)
        
        purchase_behavior.columns = ['total_revenue', 'avg_order_value', 'total_orders', 'total_quantity']
        
        return demo_analysis, purchase_behavior
    
    def analyze_product_performance(self):
        """Analyze product and category performance"""
        
        # Product performance
        product_perf = self.transactions_df.merge(
            self.products_df, on='product_id'
        ).groupby(['category', 'product_id']).agg({
            'amount': 'sum',
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        # Category performance
        category_perf = product_perf.groupby('category').agg({
            'amount': 'sum',
            'quantity': 'sum',
            'transaction_id': 'sum'
        }).sort_values('amount', ascending=False)
        
        return product_perf, category_perf
    
    def customer_lifetime_value(self):
        """Calculate Customer Lifetime Value (CLV)"""
        
        # Calculate CLV components
        clv_data = self.transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean'],
            'transaction_id': 'count',
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        clv_data.columns = ['customer_id', 'total_spent', 'avg_order_value', 'frequency', 'first_purchase', 'last_purchase']
        
        # Calculate customer lifespan in days
        clv_data['lifespan_days'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days
        clv_data['lifespan_days'] = clv_data['lifespan_days'].fillna(0)
        
        # Simple CLV calculation: Average Order Value * Purchase Frequency * Customer Lifespan
        clv_data['purchase_frequency'] = clv_data['frequency'] / (clv_data['lifespan_days'] + 1) * 365
        clv_data['clv'] = clv_data['avg_order_value'] * clv_data['purchase_frequency'] * (clv_data['lifespan_days'] / 365 + 1)
        
        return clv_data
    
    def customer_clustering(self):
        """Perform customer clustering based on behavior"""
        
        # Prepare features for clustering
        clustering_data = self.transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean'],
            'transaction_id': 'count',
            'transaction_date': lambda x: (datetime.now() - x.max()).days
        }).reset_index()
        
        clustering_data.columns = ['customer_id', 'total_spent', 'avg_spent', 'frequency', 'recency']
        
        # Standardize features
        features = ['total_spent', 'avg_spent', 'frequency', 'recency']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data[features])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clustering_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Label clusters
        cluster_labels = {0: 'Low Value', 1: 'Medium Value', 2: 'High Value', 3: 'VIP'}
        clustering_data['cluster_label'] = clustering_data['cluster'].map(cluster_labels)
        
        return clustering_data
    
    def generate_insights_report(self):
        """Generate comprehensive customer behavior insights"""
        
        print("=" * 60)
        print("CUSTOMER BEHAVIOR ANALYSIS REPORT")
        print("=" * 60)
        
        # RFM Analysis
        rfm = self.calculate_rfm_metrics()
        print("\n1. RFM SEGMENT DISTRIBUTION:")
        print(rfm['segment'].value_counts())
        
        # Customer Segments
        demo, purchase = self.analyze_customer_segments()
        print("\n2. CUSTOMER SEGMENT ANALYSIS:")
        print("Demographics:")
        print(demo)
        print("\nPurchase Behavior:")
        print(purchase)
        
        # Product Performance
        product_perf, category_perf = self.analyze_product_performance()
        print("\n3. TOP PERFORMING CATEGORIES:")
        print(category_perf.head())
        
        # CLV Analysis
        clv = self.customer_lifetime_value()
        print(f"\n4. CUSTOMER LIFETIME VALUE:")
        print(f"Average CLV: ${clv['clv'].mean():.2f}")
        print(f"Top 10% CLV: ${clv['clv'].quantile(0.9):.2f}")
        
        # Clustering
        clusters = self.customer_clustering()
        print("\n5. CUSTOMER CLUSTERS:")
        print(clusters['cluster_label'].value_counts())
        
        return rfm, demo, purchase, product_perf, category_perf, clv, clusters
    
    def create_visualizations(self):
        """Create visualizations for customer behavior analysis"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Behavior Analysis Dashboard', fontsize=16)
        
        # 1. Customer Age Distribution
        axes[0,0].hist(self.customers_df['age'], bins=20, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Customer Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Count')
        
        # 2. Revenue by Customer Segment
        segment_revenue = self.transactions_df.merge(
            self.customers_df[['customer_id', 'customer_segment']], 
            on='customer_id'
        ).groupby('customer_segment')['amount'].sum()
        
        axes[0,1].bar(segment_revenue.index, segment_revenue.values, color=['gold', 'lightgreen', 'lightcoral'])
        axes[0,1].set_title('Revenue by Customer Segment')
        axes[0,1].set_ylabel('Revenue ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Product Category Performance
        category_data = self.transactions_df.merge(self.products_df, on='product_id')
        category_revenue = category_data.groupby('category')['amount'].sum().sort_values(ascending=True)
        
        axes[0,2].barh(category_revenue.index, category_revenue.values, color='lightsteelblue')
        axes[0,2].set_title('Revenue by Product Category')
        axes[0,2].set_xlabel('Revenue ($)')
        
        # 4. Monthly Transaction Trends
        self.transactions_df['month'] = self.transactions_df['transaction_date'].dt.to_period('M')
        monthly_transactions = self.transactions_df.groupby('month').size()
        
        axes[1,0].plot(range(len(monthly_transactions)), monthly_transactions.values, marker='o', color='orange')
        axes[1,0].set_title('Monthly Transaction Trends')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Number of Transactions')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Payment Method Distribution
        payment_dist = self.transactions_df['payment_method'].value_counts()
        axes[1,1].pie(payment_dist.values, labels=payment_dist.index, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Payment Method Distribution')
        
        # 6. Customer Lifetime Value Distribution
        clv_data = self.customer_lifetime_value()
        axes[1,2].hist(clv_data['clv'], bins=30, alpha=0.7, color='mediumpurple')
        axes[1,2].set_title('Customer Lifetime Value Distribution')
        axes[1,2].set_xlabel('CLV ($)')
        axes[1,2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CustomerBehaviorAnalyzer()
    
    # Generate sample data
    analyzer.generate_sample_data(n_customers=1000, n_products=50, n_transactions=5000)
    
    # Generate comprehensive analysis report
    results = analyzer.generate_insights_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # You can also access individual analysis components:
    # rfm_data = analyzer.calculate_rfm_metrics()
    # clv_data = analyzer.customer_lifetime_value()
    # clusters = analyzer.customer_clustering()