import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

object App {
  def main(): Unit = {
    val numWorkers = sc.getExecutorMemoryStatus.size -1
    println("execute on " + numWorkers + " nodes")
    //exoMono(5, 1)
    //exoMulti(5)
    demo(6)
  }

  def demo(nbPartitions: Int): Unit = {
    val file = "demo.csv"
    val trainFile = "dota2Train75000.csv"
    println("Will read file : ./data/" + file)
    
    println("reading...")
    val train = sc.textFile("data/"+trainFile,nbPartitions)
    val parsedTrain = train.map { line =>
      val parts = line.split(',').map(_.toDouble) 
         LabeledPoint(parts(0)*0.5+0.5, Vectors.dense(parts.tail).toSparse) // *0.5+0.5 to avoid negative values
    }
    val test = sc.textFile("data/"+file)
    val parsedTest = test.map { line =>
      val parts = line.split(',').map(_.toDouble)
        LabeledPoint(-10, Vectors.dense(parts.tail).toSparse) // we set a fake label 
    }



    println("learning...")
    val model = DecisionTree.train(parsedTrain, Classification, Gini, 20)


    println("Predicting...")
    val predic = model.predict(parsedTest.first().features)
    val result = (predic-0.5)*2
    print(result)
  }

  def exoMulti(nbIte: Int): Unit = {
    val nbCores = Array(1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32)
    val file = "dota2Train.csv"

    for(nbCore <- nbCores) {
      // Load and parse the data file
      val train = sc.textFile("data/"+file,nbCore)
      val parsedTrain = train.map { line =>
        val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(0)*0.5+0.5, Vectors.dense(parts.tail).toSparse) // *0.5+0.5 to avoid negative values
      }

      // One empty training to warn up
      var model = DecisionTree.train(parsedTrain, Classification, Gini, 20)

      // BUILD MODEL
      val t0 = System.nanoTime()
      for(i <- 1 to nbIte) {
        model = DecisionTree.train(parsedTrain, Classification, Gini, 20)
      }
      val t1 = System.nanoTime()
      println(nbCore + "cores : " + ((t1 - t0)/1000000/nbIte) + "ms")

    }
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


    println("(size data / size test), (dt learn), (dt test), (Mean Squared Error )")
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





