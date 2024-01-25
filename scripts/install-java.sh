#!/bin/bash

apt-get update -y
apt install -y openjdk-11-jdk-headless
java --version
javac --version