#!/bin/bash

apt-get update -y
apt install -y openjdk-11-jdk-headless
java --version
javac --version

# install mvn javatuples
mkdir /usr/multiple && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar