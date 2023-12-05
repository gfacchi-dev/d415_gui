# D435 GUI for Face Reconstruction

This project contains a GUI for managing a couple of Intel RealSense D435 in stereo configuration, with a 33cm baseline and 30° convergence angle.

## Authors

- [@gfacchi-dev](https://github.com/gfacchi-dev)
- [@jacobuu](https://github.com/jacobuu)

## Run Locally

Clone the project

```bash
  git clone https://github.com/gfacchi-dev/d435_gui.git
```

Go to the project directory

```bash
  cd d435_gui
```

Install dependencies

```bash
  conda create --name <env> --file requirements.txt
```

Activate conda environment

```bash
  conda activate <env>
```

Start the GUI

```bash
  python ./main.py
```

## Commands

- <kbd>c</kbd>/<kbd>C</kbd> -> Calibrate the system
- <kbd>a</kbd>/<kbd>A</kbd> -> Single-shot acquisition
- <kbd>m</kbd>/<kbd>M</kbd> -> Single-shot acquisition with mesh creation
