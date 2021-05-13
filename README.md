# DocParser BentoML Serving
Serving docparser (https://github.com/lorenzo-park/DocParser) for pdf parsing.

## Usage
Please prepare **two virtual environments (each for model conversion & serving)** and it is highly recommend to use Anaconda or virtualenvs, since there are conflicts. In my guess, the reason is the docparser model is built and converted on `tensorflow==1.x` while bentoml is assuming to work on `tensorflow==2.x`. So, serving can be done in python 3.7, but model conversion should be done in python 3.6

### Pre-requisites - H5 Model Conversion to SavedModel format
1. Create an environment of python 3.6 version.
2. Install docparser
    - Install a modified version https://github.com/lorenzo-park/DocParser, a fork of the original work https://github.com/DS3Lab/DocParser with some additional functions.
    ```bash
    pip install git+https://github.com/lorenzo-park/DocParser
    ```
3. Prepare Pretrained model
    - Download from URL: https://drive.google.com/file/d/1Hi4-tg4Zmtx8zYiCg6IBi47R88PdmAW4/view?usp=sharing (Provided by the original repo https://github.com/DS3Lab/DocParser)
    - Make `model` directory and extract `docparser.h5` in the directory.

4. Set-up the right PATH and DIR varaibles in config.py
5. Freeze .h5 file and convert to `SavedModel`
    ```bash
    python convert/convertor.py
    ```

### Serving
1. Create an environment for python 3.7 version.
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Run `save.ipynb` script for registering bentoml service.
4. Run serving cli
    ```bash
    bentoml serve docparser:latest
    ```
