import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.decomposition import FactorAnalysis, PCA

app = Flask(__name__)


df = pd.read_csv(r"diabetes2.csv")  


def impute_missing_values(df):
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col].replace(0, df[col].mean(), inplace=True)
    return df

df_cleaned = impute_missing_values(df.copy())  

X = df_cleaned[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df_cleaned['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

kmeans = KMeans(n_clusters=2, random_state=42)
df_cleaned['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

fa = FactorAnalysis(n_components=2)

pca = PCA(n_components=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = ""
    logistic_regression_plot_img = ""  
    if request.method == 'POST':
      
        glucose = request.form.get('glucose', '0')
        blood_pressure = request.form.get('blood_pressure', '0')
        skin_thickness = request.form.get('skin_thickness', '0')
        insulin = request.form.get('insulin', '0')
        bmi = request.form.get('bmi', '0')
        diabetes_pedigree_function = request.form.get('diabetes_pedigree_function', '0')
        age = request.form.get('age', '0')

        
        if glucose:
            glucose = float(glucose)
        if blood_pressure:
            blood_pressure = float(blood_pressure)
       

        user_data = {
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree_function,
            'Age': age
        }

        user_input_scaled = scaler.transform(pd.DataFrame([user_data]))  # Scale user data
        predicted_outcome = model.predict(user_input_scaled)

        if predicted_outcome[0] == 1:
            prediction_result = "Predicted Outcome: Diabetic"
        else:
            prediction_result = "Predicted Outcome: Non-Diabetic"

        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], y_train, color='blue', label='Actual')  # Plot actual data points
        plt.scatter(X_train[:, 0], model.predict_proba(X_train)[:, 1], color='red', label='Predicted Probabilities')
        plt.xlabel('Glucose (Scaled)')
        plt.ylabel('Probability of Diabetes')
        plt.title('Logistic Regression: Predicted Probabilities vs. Glucose')
        plt.legend()

       
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        logistic_regression_plot_img = f'data:image/png;base64,{plot_url}'
        plt.close()

    return render_template('prediction.html',
                           prediction_result=prediction_result,
                           logistic_regression_plot_img=logistic_regression_plot_img)

@app.route('/t-test')
def t_test():
    diabetic_glucose = df_cleaned[df_cleaned['Outcome'] == 1]['Glucose']
    non_diabetic_glucose = df_cleaned[df_cleaned['Outcome'] == 0]['Glucose']
    t_stat, p_value = ttest_ind(diabetic_glucose, non_diabetic_glucose)
    t_test_result = f"T-Statistic: {t_stat}\nP-Value: {p_value}"
    return render_template('t_test.html', t_test_result=t_test_result)

@app.route('/factor_analysis', methods=['GET', 'POST'])
def factor_analysis():
    factor_analysis_plot_img = ""
    if request.method == 'POST':
        
        fa = FactorAnalysis(n_components=2) 
        fa_results = fa.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(fa_results[:, 0], fa_results[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.title('Factor Analysis')
        plt.colorbar(label='Outcome')

        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        factor_analysis_plot_img = f'data:image/png;base64,{plot_url}'
        plt.close()

    return render_template('factor_analysis.html', factor_analysis_plot_img=factor_analysis_plot_img)

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_route():
    cluster_plot_img = ""
    if request.method == 'POST':
        
        cluster_column = 'KMeans_Cluster'
        cluster_centers = kmeans.cluster_centers_  

        plt.figure(figsize=(8, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df_cleaned[cluster_column], cmap='viridis', edgecolor='k')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=300, c='red', label='Cluster Centers')
        plt.xlabel('Glucose')
        plt.ylabel('Blood Pressure')
        plt.title('KMeans Clustering')
        plt.colorbar(label='Cluster')
        plt.legend()

       
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        cluster_plot_img = f'data:image/png;base64,{plot_url}'
        plt.close()

    return render_template('kmeans.html', cluster_plot_img=cluster_plot_img)

@app.route('/pca', methods=['GET', 'POST'])
def pca_route():
    pca_plot_img = ""
    if request.method == 'POST':
   
        pca_results = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA')
        plt.colorbar(label='Outcome')

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        pca_plot_img = f'data:image/png;base64,{plot_url}'
        plt.close()

    return render_template('pca.html', pca_plot_img=pca_plot_img)

@app.route('/cleaned_dataset')
def cleaned_dataset():
  
    dataset_head = df_cleaned.head().to_html()
    return render_template('cleaned_dataset.html', dataset_head=dataset_head)

@app.route('/hierarchical', methods=['GET', 'POST'])
def hierarchical_clustering():
    hierarchical_plot_img = ""
    if request.method == 'POST':
       
        plt.figure(figsize=(10, 7))
        dendrogram(linkage(X_scaled, method='ward'), orientation='top',
                  distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')

      
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        hierarchical_plot_img = f'data:image/png;base64,{plot_url}'
        plt.close()

    return render_template('hierarchical.html', hierarchical_plot_img=hierarchical_plot_img)

if __name__ == "__main__":
    app.run(debug=True)