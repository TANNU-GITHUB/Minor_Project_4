# 🥊 Tekken 3: Touchless Computer Vision Controller

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tracking-green)

Who needs a PlayStation controller when you have a webcam? 🎮 

This project is a completely touchless, Computer Vision-powered controller that allows two players to battle it out in **Tekken 3** using physical gestures. It was built as **Project 04/12** of my *"Jab Jago Tab savera"* monthly AI/ML series, focusing on real-time tracking, pose estimation, and pure retro nostalgia.


---

## ✨ Features
* **Two-Player Support:** Splits a single webcam feed to track two players side-by-side independently.
* **Head Movement Tracking:** Uses MediaPipe FaceMesh to translate head yaw and pitch into directional movements.
* **Attack Tracking:** Uses MediaPipe Hands to track wrist velocity for rapid punches and kicks.
* **Zero Lag:** Injects keystrokes directly into the OS using `pynput` for real-time emulator response.

---

## 🕹️ Gestures & Key Bindings

To play the game, replicate these physical gestures. The script maps them to specific keyboard keys, which are then passed to the DuckStation emulator.

| Physical Gesture | In-Game Action | Player 1 Key | Player 2 Key |
| :--- | :--- | :---: | :---: |
| **Head Tilt Left** | Walk Left | `A` | `Left Arrow` |
| **Head Tilt Right** | Walk Right | `D` | `Right Arrow` |
| **Head Look Up** | Jump | `W` | `Up Arrow` |
| **Head Look Down** | Crouch | `S` | `Down Arrow` |
| **Left Hand Fast Swipe** | Square (Punch) | `F` | `J` |
| **Right Hand Fast Swipe** | Cross (Kick) | `G` | `K` |
| **Left Hand Downward Swipe**| Triangle | `T` | `I` |
| **Right Hand Downward Swipe**| Circle | `H` | `L` |

---

## 🚀 How to Run It on Your Laptop

Follow these steps to set up the environment and start playing:

### 1. Prerequisites
* **Python 3.8+** installed on your system.
* A working webcam.
* **DuckStation** (or any PS1 emulator) and a Tekken 3 ROM.

### 2. Configure Your Emulator
Open DuckStation and map your controllers exactly to the keys listed in the table above:
* **Controller 1 (Player 1):** Map D-Pad to `W/A/S/D` and Face Buttons to `F/G/H/T`.
* **Controller 2 (Player 2):** Map D-Pad to `Up/Left/Down/Right` and Face Buttons to `J/K/L/I`.

### 3. Installation
Clone this repository to your local machine:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
