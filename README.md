# 📱 Tor-Based Website Fingerprinting (WF) Attack on Android

This repository demonstrates a Website Fingerprinting (WF) attack targeting Android applications using Tor.

---

## 1. TorCollect: Android Data Collection App

- **TorCollect** is an Android app that utilizes the `TrafficStats` API.
- It works with `server.py` to collect upload/download traffic statistics from the device.
- To simulate WF attacks on Tor users, we use the **Orbot** app to route all Android traffic through Tor.

---

## 2. How to Collect Tor Traffic Data

To collect Tor traffic data from your Android device, follow these steps:

1. **Set up Tor on your computer**
    - Create a `torrc` file with the following lines:
      ```
      HiddenServiceDir /path/to/hidden_service/
      HiddenServicePort 80 127.0.0.1:5001
      ```
    - This will generate a `.onion` address inside the `hostname` file in the specified directory.

2. **Start Tor on your computer**
    - Make sure the hidden service is running.

3. **Configure your Android device**
    - Install and launch **Orbot**.
    - Ensure it routes traffic through the Tor network.

4. **Run the server and client**
    - Run `server.py` on your computer.
    - Launch the **TorCollect** app on your Android device.

---

## 3. Data Collection

- The app will send upload/download traffic data to the server via the Tor network using the generated `.onion` address.

---

## 4. Data Preprocessing

1. Run `raw_to_diff.py`  
   → Converts cumulative traffic data into difference-based sequences.

2. Run `packet_seq_augmentation.py`  
   → Converts data into a **per-packet** format and augment the datas on packet sequences.

3. Run `attach_noise.py`  
   → Appends random noise and adjusts sequence lengths to a fixed size.

---

## 5. Model Evaluation

- Use `model.py` (MLP-based) to train and evaluate the WF attack model on the processed traffic data.

---

## ⚠️ Note

- Ensure your `.onion` hostname and private key files are **not** uploaded to public repositories.
- All experiments assume traffic is routed through Tor and accessed via hidden services.
