import org.apache.avro.generic.GenericData;
import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args){

        int nbFichier = 8;
        long[] resultsTime = new long[nbFichier];
        List<String> logFile = new ArrayList<String>(Arrays.asList(
                "./superconduct/train_200.csv",
                "./superconduct/train_500.csv",
                "./superconduct/train_1000.csv",
                "./superconduct/train_2500.csv",
                "./superconduct/train_5000.csv",
                "./superconduct/train_10000.csv",
                "./superconduct/train_20000.csv",
                "./superconduct/train_40000.csv"));
        SparkConf conf = new SparkConf().setAppName("Test App").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        for(int j = 0 ; j < nbFichier ; ++j){
            JavaRDD<String> data  = sc.textFile(logFile.get(j)).cache();
            JavaRDD<Vector> parsedData = data.map(
                    new Function<String, Vector>() {
                        public Vector call(String s) {
                            String[] sarray = s.split(",");
                            double[] values = new double[sarray.length];
                            for (int i = 0; i < 2; i++)
                                values[i] = Double.parseDouble(sarray[i]);
                            return Vectors.dense(values);
                        }
                    }
            );
            parsedData.cache();

            // Cluster the data into two classes using KMeans
            int numClusters = 10;
            int numIterations = 30;
            int nbIte = 20;
            long startTime = System.currentTimeMillis();
            for(int i = 0 ; i < nbIte ; ++i)
                KMeans.train(parsedData.rdd(), numClusters, numIterations);
            long endTime = System.currentTimeMillis();
            resultsTime[j]=(endTime-startTime)/nbIte;
        }

        for(int j = 0 ; j < nbFichier ; ++j) {
            System.out.println(logFile.get(j).replace("./superconduct/train_","").replace(".csv","")+" " + resultsTime[j]);
        }

    }
}