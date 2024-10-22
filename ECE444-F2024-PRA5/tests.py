import pytest
import requests

url = "http://fake-news-det-env.eba-i28ffpzg.us-east-2.elasticbeanstalk.com/predict"

test_cases = [
    ("This has to be fake.", "FAKE"),
    ("Completely false information.", "FAKE"),
    ("The president made an official statement today.", "REAL"),
    ("Biden won the 2024 election.", "REAL")
]

def run_tests():
    passed = 0
    failed = 0
    results = []

    # Run each test case
    for idx, (input_text, expected_output) in enumerate(test_cases):
        try:
            data = {"text": input_text}
            response = requests.post(url, json=data)

            # Ensure the request was successful
            if response.status_code != 200:
                results.append(f"Test Case {idx + 1}: Failed (Status Code: {response.status_code})")
                failed += 1
                continue
            
            # Check if the returned prediction matches the expected output
            prediction = response.json().get('prediction')
            if prediction == expected_output:
                results.append(f"Test Case {idx + 1}: Passed")
                passed += 1
            else:
                results.append(f"Test Case {idx + 1}: Failed (Expected: {expected_output}, Got: {prediction})")
                failed += 1
        except Exception as e:
            results.append(f"Test Case {idx + 1}: Failed (Exception: {str(e)})")
            failed += 1

    # Output the test results
    print("\nTest Results:\n")
    for result in results:
        print(result)

    print("\nSummary:")
    print(f"Total Passed: {passed}")
    print(f"Total Failed: {failed}")

if __name__ == "__main__":
    run_tests()