import os
import csv
from datetime import datetime
import pytz
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import re

app = Flask(__name__)

traffic_data_log = []
LOG_DIR = "tor/"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def clean_url(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or url
    clean_name = re.sub(r'^www\.', '', hostname)
    clean_name = re.sub(r'\.(com|org|net|io|gov|edu|co|info|biz|us|uk|jp|kr|cn|in|ru|de|fr|it|es|ca|au|br|mx|nl|se|no|fi|za|sg|my|hk|tw|vn|th|ph|id)$', '', clean_name)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_name)
    return clean_name

def save_logs_to_file(url):
    if not url:
        print("Error: URL is missing")
        return

    clean_name = clean_url(url)
    folder_path = os.path.join(LOG_DIR, clean_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if traffic_data_log:
        # making file name with current time.
        current_time = datetime.now(pytz.timezone("Asia/Seoul"))
        timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
        filename = f"{clean_name}_{timestamp_str}.csv"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "uploadbytes", "downloadbytes", "uploadpackets", "downloadpackets", "flag"])
            writer.writerows([[log["timestamp"], log["uploadbytes"], log["downloadbytes"], 
                              log["uploadpackets"], log["downloadpackets"], log["flag"]] for log in traffic_data_log])

        print(f"Logs saved to {file_path} with {len(traffic_data_log)} entries")
        traffic_data_log.clear()
    else:
        print("traffic_data_log doesn't exist!")

@app.route('/save', methods=['GET'])
def save():
    url = request.args.get('url', type=str)
    if not url:
        return jsonify({"status": "error", "message": "Missing 'url' parameter"}), 400

    save_logs_to_file(url)
    return "Log File Updated"

@app.route('/jsonify_traffic', methods=['POST'])
def jsonify_traffic():

    data = request.json
    if not isinstance(data, list):
        print("Error: Expected a list of data points")
        return jsonify({"status": "error", "message": "Expected a list of data points"}), 400

    for item in data:
        timestamp_ms = item.get('timestamp_ms')
        uploadbytes = item.get('uploadbytes')
        downloadbytes = item.get('downloadbytes')
        uploadpackets = item.get('uploadpackets')
        downloadpackets = item.get('downloadpackets')
        flag = item.get('flag')
        if None in (timestamp_ms, uploadbytes, downloadbytes, uploadpackets, downloadpackets, flag):
            continue
        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000.0, pytz.timezone("Asia/Seoul"))
        timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        traffic_data_log.append({
            "timestamp": timestamp_str,
            "uploadbytes": uploadbytes,
            "downloadbytes": downloadbytes,
            "uploadpackets": uploadpackets,
            "downloadpackets": downloadpackets,
            "flag": flag
        })

    print(f"Received batch data with {len(data)} entries")
    return jsonify({"status": "success", "message": f"Received {len(data)} data points"})

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify({"status": "success", "data": traffic_data_log})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
