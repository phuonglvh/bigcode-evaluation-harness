#!/bin/bash

apt-get update -y
apt install -y openjdk-11-jdk-headless
java --version
javac --version

EVAL_JAVA_EXTRA_CLASSPATH_FOLDER="${EVAL_JAVA_EXTRA_CLASSPATH_FOLDER:-'/usr/multiple'}"

# install mvn javatuples
mkdir -p "$EVAL_JAVA_EXTRA_CLASSPATH_FOLDER" && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O "$EVAL_JAVA_EXTRA_CLASSPATH_FOLDER/javatuples-1.2.jar"