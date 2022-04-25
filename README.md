# Embedded CNN

This repository presents the results of a study designed to assess the  performance of Convolutional Neural Networks (CNN) on embedded devices. The test is performed on deep learning models, containing many convolutions in  hidden layers as SSD, YOLO, etc.

## Usage

On terminal, copy:

`git clone https://github.com/jpfcabral/embedded-cnn.git `

Then, install requirements:

`cd embedded-cnn`


`pip install -r requirements.txt `

(Optional) Install raspbbery module to verify temperature

`pip install gpiozero`

Finally, execute tests:

`pytest ./tests -vs`

## TODO

- Mobilenetv2 Implementation
- Test on Raspberry Pi 3

## Tested devices

|Device          |SoC             |CPU                |RAM (MB)|GPU                          |
|----------------|----------------|-------------------|--------|-----------------------------|
|Raspberry Pi 1B+|Broadcom BCM2835|1x ARM 700 MHz     |512     |Broadcom VideoCore IV 250 MHz|
|Raspberry Pi 2B |Broadcom BCM2836|4x Cortex-A7 900 MHz|1024    |Broadcom VideoCore IV 250 MHz|
