//ulimit -n 4096
//spark-shell --packages com.databricks:spark-csv_2.11:1.4.0
//IP: 10.60.37.85



import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Calendar
import org.apache.spark.sql.types.DoubleType



val renewal = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("HWIN_RENEWAL_HISTORY_Simplified.csv")

val r = renewal.select(renewal("COMPANY_ID"), renewal("CURRENT_YEAR"), $"DATE_RENEWAL_COMPLETE").distinct()


val isExpired = udf ((currentYear:String, renewDate:String) => 
     if (renewDate == null||renewDate.isEmpty()) 1 else { 
         val year =  currentYear.substring(currentYear.length() -2)
         val sdf = new SimpleDateFormat("dd-MMM-yy")
         val cDate = sdf.parse(currentYear)
         val rDate = sdf.parse(renewDate)
         val calendar =Calendar.getInstance 
         calendar.setTime(cDate)
         val cyear = calendar.get(Calendar.YEAR)
         calendar.set(cyear,  Calendar.FEBRUARY, 15)
	 val deadlineDate = calendar.getTime
         if (rDate.after(deadlineDate)) 1
         else 0
     }
)
         
 
val r1 = r.withColumn("EXPIRED", isExpired(r("CURRENT_YEAR"), r("DATE_RENEWAL_COMPLETE")))

val r2015=r1.filter($"CURRENT_YEAR".endsWith("10")||$"CURRENT_YEAR".endsWith("11")||$"CURRENT_YEAR".endsWith("12")||$"CURRENT_YEAR".endsWith("13")||$"CURRENT_YEAR".endsWith("14")||$"CURRENT_YEAR".endsWith("15")).groupBy("COMPANY_ID").sum("EXPIRED")


val manifests = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("2002_2016_MANIFESTS_2015.csv")

val m1 = manifests.select($"COMPANY_ID", $"DATE_SHIPPED_YEAR", $"MANIFEST_FEE", $"TONNAGE_FEE").filter("DATE_SHIPPED_YEAR = 2015")
val m2 = m1.withColumn("total", $"MANIFEST_FEE" + $"TONNAGE_FEE")

val m3 = m2.groupBy("COMPANY_ID").sum("total")
val m4 = m2.groupBy("COMPANY_ID").count
val m5 = m3.join(m4, "COMPANY_ID")
val r2016=r1.filter($"CURRENT_YEAR".endsWith("16")).select($"COMPANY_ID", $"EXPIRED")
//val m6 = m5.join(r2016, "COMPANY_ID")
val m6 = r2016.join(m5, "COMPANY_ID").join(r2015, "COMPANY_ID")


val lr1 = m6.withColumnRenamed("EXPIRED", "label")
val lr2 = lr1.withColumn("label", lr1.col("label").cast(DoubleType))

val assembler = new VectorAssembler().setInputCols(Array("sum(EXPIRED)", "count")).setOutputCol("features")

val vectorDf = assembler.transform(lr2)
//val vectorDf1 = vectorDf.select($"COMPANY_ID", $"label", $"features");
//val lrModel = new LogisticRegression().setMaxIter(10).setRegParam(0).setElasticNetParam(0)
val lrModel = new LogisticRegression().setMaxIter(10).setRegParam(0)
//val lrDF = vectorDf.select($"label", $"features")


val Array(trainingData, testData) = vectorDf.randomSplit(Array(0.7, 0.3))
val lrmodel = lrModel.fit(trainingData)
lrmodel.transform(testData).filter("prediction = label").count
lrmodel.transform(testData).filter("prediction = label and prediction = 1").count
testDate.filter("label=1").count

val nbModel = new NaiveBayes
val nbmodel = nbModel.fit(trainingData)
nbmodel.transform(testData).filter("prediction = label").count
nbmodel.transform(testData).filter("prediction = label and prediction = 1").count


//evaluate
import org.apache.spark.mllib.evaluation.{RegressionMetrics, MulticlassMetrics}

 val fullPredictions = nbmodel.transform(trainingData).cache()
    val predictions = fullPredictions.select("prediction").map(_.getDouble(0))
    val labels = fullPredictions.select("label").map(_.getDouble(0))
  


val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    println(s"  Root mean squared error (RMSE): $RMSE")






val nbModel = new NaiveBayes()
