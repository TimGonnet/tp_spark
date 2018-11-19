import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

object App {
  def main(): Unit = {
    exoMono(5, 1)
  }
  def test(): Unit = {
    println("Hello, world!")
  }

  def exoMulti(nbCores: Int, nbIte: Int): Unit = {

  }

  def exoMono(nbIte: Int, nbPartitions: Int): Unit = {
    val files = Array("dota2Train5000.csv", 
      "dota2Train10000.csv", 
      "dota2Train25000.csv", 
      "dota2Train37500.csv", 
      "dota2Train50000.csv", 
      "dota2Train62500.csv", 
      "dota2Train75000.csv", 
      "dota2Train.csv");


    println(file + " : (size data / size test), (dt learn), (dt test), (Mean Squared Error )")
    for(file <- files){
      // READ FILE
      // Load and parse the data file
      val train = sc.textFile("data/"+file,nbPartitions)
      val parsedTrain = train.map { line =>
        val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(0)*0.5+0.5, Vectors.dense(parts.tail).toSparse) // *0.5+0.5 to avoid negative values
      }
      val test = sc.textFile("data/dota2Test.csv")
      val parsedTest = test.map { line =>
        val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(0)*0.5+0.5, Vectors.dense(parts.tail).toSparse)
      }

      // One empty training to warn up
      var model = DecisionTree.train(parsedTrain, Classification, Gini, 20)

      // BUILD MODEL
      val t0 = System.nanoTime()
      for(i <- 1 to nbIte) {
        model = DecisionTree.train(parsedTrain, Classification, Gini, 20)
      }
      val t1 = System.nanoTime()

      // Evaluate model on training examples and compute training error

      // One empty training to warn up
      var valuesAndPreds = parsedTest.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }

      val t2 = System.nanoTime()
      for(i <- 1 to nbIte) {
        valuesAndPreds = parsedTest.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
      }
      val t3 = System.nanoTime()
  
      print( "(" + parsedTrain.count() + " / " + parsedTest.count() + "), ")
      print( ((t1 - t0)/1000000/nbIte) + "ms, ")
      print( ((t3 - t2)/1000000/nbIte) + "ms, ")
      val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.mean()
      println(MSE)
    }
  }
}





