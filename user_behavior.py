import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------- Task 1 ----------------------------

# Βασισμένο στα PDF 4 & 6 (σελ. 32 "Load Dataset & Clean")
def load_and_clean_data(path):
    try:
        # Φόρτωση με standard pandas read_csv (όπως στα παραδείγματα των PDF)
        df = pd.read_csv(path)

        # Καθαρισμός: Αφαίρεση διπλότυπων και κενών τιμών (dropna)
        # Στο PDF 6 (σελ. 32) χρησιμοποιείται το dropna για καθαρισμό
        df = df.drop_duplicates()
        df = df.dropna()

        return df
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση: {e}")
        return None


# ------------------- Task 2 ----------------------------

class UserMetrics:
    def __init__(self, user_data):
        self.user_data = user_data

    def activity_score(self):
        return self.user_data["sessions"] * 2

    def engagement_score(self):
        return self.user_data["avg_session_time"] * self.user_data["sessions"]

    def purchase_score(self):
        return self.user_data["purchases"] * 5

    def inactivity_penalty(self):
        return self.user_data["last_active_days"] * 1.5

    def total_score(self):
        return (self.activity_score() + self.engagement_score() + self.purchase_score() - self.inactivity_penalty()
    )

class UserProfiler:
    def __init__(self, score):
        self.score = score

    def get_profile(self):
        if self.score >= 500:
            return "Highly Active User"
        elif self.score >= 200:
            return "Moderately Active User"
        else:
            return "Inactive User"


def final_user_scores(df):
    scores = []
    profiles = []

    for i in range(len(df)):
        row = df.iloc[i]
        metrics = UserMetrics(row)
        score = metrics.total_score()
        profiler = UserProfiler(score)
        profile = profiler.get_profile()
        scores.append(score)
        profiles.append(profile)

    df["user_score"] = scores
    df["user_profile"] = profiles
    profile_stats = df["user_profile"].value_counts()

    print("User Profile distribution:")
    print(profile_stats)
    print("\nSample Results:")
    print(df[["user_id", "user_score", "user_profile"]].head(20))


# ------------------- Task 3 ----------------------------

# Business Logic (Ομαδοποίηση Χρηστών)
# Βασισμένο στο PDF 6 (σελ. 17 "Sentiment Analysis Logic")
def define_user_group(row):
    """
    Βοηθητική συνάρτηση που εφαρμόζεται σε κάθε γραμμή (row).
    Αντιστοιχεί στη λογική 'def clean_text' ή 'lambda' των διαφανειών.
    """
    if row['purchases'] > 10:
        return 'VIP'
    elif row['sessions'] > 50:
        return 'Active'
    else:
        return 'Standard'


def assign_user_groups(df):
    # Χρήση της apply με axis=1 για έλεγχο σε κάθε γραμμή
    # Αυτή είναι η μέθοδος που διδάσκεται στο PDF 6 για conditional logic
    df['user_group'] = df.apply(define_user_group, axis=1)
    return df


# Aggregations & Visualization
# Βασισμένο στα PDF 4 (GroupBy) και PDF 6 (Visualization σελ. 20, 34)
def calculate_and_plot_stats(df):
    # 3.1 GroupBy Aggregation
    # Ομαδοποίηση και υπολογισμός στατιστικών
    stats = df.groupby('user_group').agg({
        'user_id': 'count',  # Count users
        'sessions': 'sum',  # Total sessions
        'avg_session_time': 'mean',  # Average time
        'purchases': 'sum'  # Total purchases
    }).reset_index()

    # Μετονομασία για σαφήνεια
    stats.columns = ['User Group', 'Total Users', 'Total Sessions', 'Avg Time (min)', 'Total Purchases']

    print("\n--- 3.2 Συγκεντρωτικά Στατιστικά ---")
    print(stats.to_string(index=False))

    # 3.2 Visualization (Matplotlib)
    # Όπως στο PDF 6 (σελ. 20, 34), κάνουμε visualize τα αποτελέσματα του grouping
    plt.figure(figsize=(10, 6))

    # Bar chart για τον αριθμό χρηστών ανά κατηγορία
    plt.bar(stats['User Group'], stats['Total Users'], color='skyblue', edgecolor='black')

    plt.title('Κατανομή Χρηστών ανά Ομάδα (VIP, Active, Standard)')
    plt.xlabel('User Group')
    plt.ylabel('Πλήθος Χρηστών')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Αποθήκευση ή εμφάνιση (όπως στα παραδείγματα)
    plt.show()

    return stats

# 1. Φόρτωση
#df_raw = load_and_clean_data(file_path)

#if df_raw is not None:
    # 2. Εφαρμογή Λογικής με apply()
    #df_grouped = assign_user_groups(df_raw)

    # 3. Υπολογισμός και Γράφημα
    #final_stats = calculate_and_plot_stats(df_grouped)

    # Προαιρετικά: Εξαγωγή σε CSV (PDF 5/6 practice)
    # final_stats.to_csv('group_stats.csv', index=False)

# ------------------- Task 4 ----------------------------

# Οπτικοποίηση κατανομών
def churn_prediction(df):
    X = df[["sessions", "avg_session_time", "purchases", "last_active_days"]]
    Y = df["churned"]
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

    # Εκπαίδευση μοντέλου churn
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Προβλέψεις
    y_pred = model.predict(X_test)

    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # Υπολογισμός χρηστών με υψηλή πιθανότητα αποχώρησης
    high_risk_users = df[df["churn_probability"] > 0.7]
    print("Χρήστες με υψηλή πιθανότητα αποχώρησης:", len(high_risk_users))


    # Αξιολόγηση μοντέλου
    accuracy = accuracy_score(Y_test, y_pred)
    print("Ακρίβεια μοντέλου:", accuracy)

    # Οπτικοποίηση αποτελεσμάτων (γράφημα)
    plt.figure()
    plt.pie(df["churned"].value_counts(), labels=["Active", "Churned"], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title("Κατανομή Χρηστών Churned vs Active")
    plt.show()


# ------------------- MAIN SYNARTHSH ----------------------------
def main():
    file_path = "ex4_user_behavior_dataset.csv"
    df = load_and_clean_data(file_path)

    if df is None:
        return

    final_user_scores(df)
    df_grouped = assign_user_groups(df)  #ananewmeno df
    calculate_and_plot_stats(df_grouped)
    churn_prediction(df_grouped)


if __name__ == "__main__":
    main()