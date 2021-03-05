This is Read Me file where i will write the information about the project...

https://github.com/maxpumperla/elephas#basic-spark-integration 

Dependent library

sudo apt install openjdk-8-jre-headless 
sudo update-java-alternatives -s java-1.8.0-openjdk-amd64 
pip3 install py4j 
pip3 install elephas  

Download spark 

wget "https://mirrors.estointernet.in/apache/spark/spark-3.1.1/spark-3.1.1-bin-hadoop2.7.tgz"

install spark 
tar xzf spark-3.1.1-bin-hadoop2.7.tgz  
export SPARK_HOME = "<path-to-the-root-of-your-spark-installation>"  

sudo apt install spark  


Run the testddl using the command  
python3 testddl.py  
or  
spark-submit --driver-memory 2G ./testddl.py   

