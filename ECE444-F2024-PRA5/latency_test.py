import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_api_calls(url, data, n_calls=100):
    timestamps = []
    
    for _ in range(n_calls):
        start_time = time.time()  # Start timer
        response = requests.post(url, json=data)
        end_time = time.time()  # End timer

        # Record latency (in milliseconds)
        timestamps.append((end_time - start_time) * 1000)

    return timestamps

def main():
    url = "http://fake-news-det-env.eba-i28ffpzg.us-east-2.elasticbeanstalk.com/predict"  # replace with your actual Elastic Beanstalk URL
    test_cases = [
        ("This has to be fake.", "FAKE"),
        ("Completely false information.", "FAKE"),
        ("The president made an official statement today.", "REAL"),
        ("Biden won the 2024 election.", "REAL")
    ]

    # Store results in a dataframe
    results = pd.DataFrame(columns=['Test Case', 'Latency (ms)'])

    for case_name, data in test_cases:
        latencies = make_api_calls(url, data)
        # Append to dataframe
        for latency in latencies:
            results = pd.concat([results, pd.DataFrame([{'Test Case': case_name, 'Latency (ms)': latency}])], ignore_index=True)

    # Save to CSV
    results.to_csv('api_performance.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Test Case', y='Latency (ms)', data=results)
    plt.xticks(rotation=45, ha="right", fontsize=10) 
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.xlabel("Test Case", fontsize=12)
    plt.title("API Latency Performance", fontsize=14)
    plt.tight_layout()


    plt.show()

if __name__ == '__main__':
    main()