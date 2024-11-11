#pragma warning disable CS8604 // (Handled) Possible null reference argument. 
#pragma warning disable CS8602 // (Handled) Dereference of a possibly null reference.

using System.Diagnostics.CodeAnalysis;

using UCC124111245.ML.Classification;
using UCC124111245.Utilities;

namespace Main.App;

/// <summary>
/// This Class is the entry point for MultinomialLogisticRegression algorithm to run.
/// </summary>
/// <remarks>Author: Anish Arya</remarks>
class MultinomialLogisticRegressionApp
{
  private string? _dataFilePathUserInput;

// ----------------------------------------------------------------------

  /// <summary>
  /// This helper method does the training and testing on training and validations datasets repectively, and logs and prints the output.
  /// Also, it logs the epochs and loss in the epochs output file in the "epochsAndLossOutputViewFiles" directory.
  /// </summary>
  /// <param name="csvReader">This is a non-null and non-empty parameter of csvreader class instance.</param>
  /// <param name="epochs">This is a non-null and non-empty parameter of number of training iterations.</param>
  /// <param name="learningRate">This is a non-null and non-empty parameter of the rate at which the training will happen at each epoch.</param>
  /// <param name="temperatureForModifiedSoftmax">This is a non-null and non-empty parameter of scaling the probabilities to prevent overflow.</param>
  /// <param name="randomSeed">This is a non-null and non-empty parameter of giving seed for consistent results.</param>
  /// <param name="testSize">This is a non-null and non-empty parameter of the split of test dataset from the original dataset (between 0 and 1).</param>
  /// <param name="outputFilepath">This is a non-null and non-empty parameter of output file path for logging the epochs and loss in the epochs output file in the "epochsAndLossOutputViewFiles" directory.</param>
  /// <remarks>Author: Anish Arya</remarks>
  private void _dTrainAndTest(
    [DisallowNull] ClassificationCSVReader csvReader,
    [DisallowNull] int[] epochs,
    [DisallowNull] double learningRate,
    [DisallowNull] int temperatureForModifiedSoftmax,
    [DisallowNull] int randomSeed,
    [DisallowNull] double testSize,
    [DisallowNull] string outputFilepath)
  {
    // creating an instance of MultinomialLogisticRegression class
    MultinomialLogisticRegression mlr = new MultinomialLogisticRegression(
      numberOfPredictorFeatures: csvReader.NumTotalFeatures - 1, 
      numberOfClassesInTargetFeature: csvReader.ClassesOfTargetFeature.Count, 
      learningRate: learningRate);

    // one-hot concoding the traget class
    double[,] yOneHot = HelperProcessData.OneHotEncode(
      numDataXis: csvReader.NumDataXis,
      numClassesInFeature: csvReader.ClassesOfTargetFeature.Count,
      y: csvReader.y
    );

    // stratified train-test split
    int numTestXis = (int)(csvReader.NumDataXis * testSize);
    int numTrainXis = csvReader.NumDataXis - numTestXis;
    int numberOfClassesInTargetFeature = csvReader.ClassesOfTargetFeature.Count;

    double[,] XTrain = new double[numTrainXis, csvReader.NumTotalFeatures - 1];
    double[,] yOneHotTrain = new double[numTrainXis, numberOfClassesInTargetFeature];
    double[,] XTest = new double[numTestXis, csvReader.NumTotalFeatures - 1];
    double[,] yOneHotTest = new double[numTestXis, numberOfClassesInTargetFeature];

    var stratifiedTrainTestSplit = HelperProcessData.StratifiedTrainTestSplit(
      X: csvReader.X,
      yOneHot: yOneHot,
      randomSeed: randomSeed,
      testSize: testSize
    );

    XTrain = stratifiedTrainTestSplit.Item1; yOneHotTrain = stratifiedTrainTestSplit.Item2;
    XTest = stratifiedTrainTestSplit.Item3;  yOneHotTest = stratifiedTrainTestSplit.Item4;
    
    // for each epoch train
        //  1. Train;
        //  2. calculate loss;
        //  3. only for desired number of epochs, Test and display confusion matrix and compute metrics.
    
    double trainingLoss = 0.0;
    int[,] confusionMatrix = 
          new int[numberOfClassesInTargetFeature, numberOfClassesInTargetFeature];

    StreamWriter writer = new StreamWriter(outputFilepath);
    try{
      writer.WriteLine("NumberOfEpochs,TrainingLoss");
      // Only total of Max of epochsList iterations will be there in the output.
      // Randomness issue has been take care of via random seed.
      // Hence, it saves computing power and time.
      for(int epochidx = 1; epochidx <= epochs.Max(); ++epochidx)
      {
        // Train
        trainingLoss = mlr.DTrain(
          XTrain: XTrain,
          yOneHotTrain: yOneHotTrain,
          temperatureForModifiedSoftmax: temperatureForModifiedSoftmax);
        writer.WriteLine($"{epochidx},{trainingLoss}");
        
        // only for desired number of epochs, Test and display confusion matrix and compute metrics.
        if(epochs.Contains(epochidx))
        {
          Console.WriteLine("*********************************************\n");
          Console.WriteLine($"Number of Epochs: {epochidx}, Training Loss: {trainingLoss}");

          // Test and display confusion matrix
          confusionMatrix = mlr.DTestAndDisplayConfusionMatrix(
                            XTest: XTest,
                            yOneHotTest: yOneHotTest,
                            classesOfTargetFeature: csvReader.ClassesOfTargetFeature
                          );
          
          mlr.PrintConfusionMatrix(confusionMatrix);

          //fetch #misclassifications and accuary
          var acc = HelperComputeMetrics.AccuracyAndMisclassifications(confusionMatrix);
          
          // print model performance metrics
          Console.WriteLine(
            $"\nModel Prformance Metrics:\n\t#Misclassifications: {acc.Item1} / " +
                  $"{XTest.GetLength(0)}\n"+
            $"\tAccuracy: {Math.Round(acc.Item2, 4)}\n" +
            $"\tPrecision: {Math.Round(HelperComputeMetrics.Precision(confusionMatrix), 4)}\n"+
            $"\tRecall: {Math.Round(HelperComputeMetrics.Recall(confusionMatrix), 4)}\n" +
            $"\tF1 Score: {Math.Round(HelperComputeMetrics.F1Score(confusionMatrix), 4)}\n");
        }
      }
    }
    finally{
      writer.Close();
    }
  }

  /// <summary>
  /// This helper method does splits the dataset in input features and output feature(with labels).
  /// </summary>
  /// <param name="csvReader">This is a non-null and non-empty parameter of csv reader class instance.</param>
  /// <remarks>Author: Anish Arya</remarks>
  private void _processData(
    [DisallowNull] ClassificationCSVReader csvReader)
  {
    csvReader.ProcessXYPairs();
  }

  /// <summary>
  /// This helper method acts the bootup for the entire process from training to output of the application.
  /// </summary>
  /// <param name="epochs">This is a non-null and non-empty parameter of list of number of training iterations at which the model parameters metrics will get calculated.</param>
  /// <param name="learningRate">This is a non-null and non-empty parameter of the rate at which the training will happen at each epoch.</param>
  /// <param name="temperatureForModifiedSoftmax">This is a non-null and non-empty parameter of scaling the probabilities to prevent overflow.</param>
  /// <param name="randomSeed">This is a non-null and non-empty parameter of giving seed for consistent results.</param>
  /// <param name="testSize">This is a non-null and non-empty parameter of the split of test dataset from the original dataset (between 0 and 1).</param>
  /// <param name="outputFilepath">This is a non-null and non-empty parameter of output file path for logging the epochs and loss in the epochs output file in the "epochsAndLossOutputViewFiles" directory.</param>
  
  /// <remarks>Author: Anish Arya</remarks>
  private void _run(
    [DisallowNull] int[] epochs,
    [DisallowNull] double learningRate,
    [DisallowNull] int temperatureForModifiedSoftmax,
    [DisallowNull] int randomSeed,
    [DisallowNull] double testSize,
    [DisallowNull] string outputFilepath
  )
  {
    // CSV reader instance
    ClassificationCSVReader csvReader = new();
    // read CSV
    try{
      csvReader.ReadCSVData(dataFilePath: this._dataFilePathUserInput);
    }
    catch(Exception ex)
    {
      if(ex is FileNotFoundException)
      {
        if(ex.Message.Contains("File not found for CSV reader."))
          Console.WriteLine($"\nBad File Path. File not found for CSV reader to read.\nFile path provided:"+
          $" {this._dataFilePathUserInput}\n\nDetailed StackTrace is below:\n\n");
        else
          Console.WriteLine("\nNo data in Input File.\n\nDetailed StackTrace is below:\n\n");
        Console.WriteLine(ex.StackTrace);
        
        Console.WriteLine();
        Console.WriteLine("Press return/enter key to exit.");
        Console.Read();
      }
      if(ex is InvalidDataException)
        Console.WriteLine("Ensure Input File format to be of type .csv");
      else
      {
        // (do not remove this else block for better exception handling)
      }
    }
    
    // process Csv
    this._processData(csvReader: csvReader);

    // train Model
    this._dTrainAndTest(
      csvReader: csvReader, 
      epochs: epochs, 
      learningRate: learningRate,
      temperatureForModifiedSoftmax: temperatureForModifiedSoftmax,
      randomSeed: randomSeed,
      testSize: testSize,
      outputFilepath: outputFilepath);

  }

  /// <summary>
  /// This constructor method sets the input csv file path.
  /// </summary>
  /// <param name="dataFilePathUserInput">This is a non-null and non-empty parameter of input csv file path.</param>
  /// <remarks>Author: Anish Arya</remarks>
  public MultinomialLogisticRegressionApp(
    [DisallowNull] string? dataFilePathUserInput) {
    this._dataFilePathUserInput = dataFilePathUserInput;
  }

  public static void Main(string[] args)
  {
    // list of number of training iterations at which the model parameters metrics will get calculated.
    int[] epochsList = new int[] {1, 10, 100, 250, 500};

    Console.WriteLine();
    Console.WriteLine("----------------------------------------------data3cat.csv----------------------------------------------");
    Console.WriteLine("--------------------------------------------------------------------------------------------------------");
    
    // get path of the current directory and set the respective input and output files
    string? file1Data3CatPath = System.IO.Path.GetFullPath(@"..//data//data3cat.csv");
    string? file1Data3CatEpochOutputPath = System.IO.Path.GetFullPath(@"..//epochsAndLossOutputViewFiles//MLogR-Data3Cat-epochsOutput.csv");
    
    MultinomialLogisticRegressionApp appData3Cat = new(file1Data3CatPath);

    // bootup the application for a dataset
    appData3Cat._run(
      epochs: epochsList,
      learningRate: 0.01,
      temperatureForModifiedSoftmax: 10,
      randomSeed: 5,
      testSize: 0.3,
      outputFilepath: file1Data3CatEpochOutputPath
    );
    Console.WriteLine("\n*********************************************");

    Console.WriteLine();

    Console.WriteLine("----------------------------------------------MLRData.csv----------------------------------------------");
    
    Console.WriteLine("-------------------------------------------------------------------------------------------------------");
    
    // get path of the current directory and set the respective input and output files
    string? file2MLRDataPath = System.IO.Path.GetFullPath(@"..//data//MLRdata.csv");
    string? file2MLRDataEpochOutputPath = System.IO.Path.GetFullPath(@"..//epochsAndLossOutputViewFiles//MLogR-MLRdata-epochsOutput.csv");
    
    MultinomialLogisticRegressionApp appMLRData = new(file2MLRDataPath);

    // bootup the application for a dataset
    appMLRData._run(
      epochs: epochsList,
      learningRate: 0.01,
      temperatureForModifiedSoftmax: 10,
      randomSeed: 5,
      testSize: 0.3,
      outputFilepath: file2MLRDataEpochOutputPath
    );
    
    Console.WriteLine("--------------------------------------------------------------------------------------------------------");

    Console.WriteLine();
  
    Console.WriteLine("Please Note 1: \n\tDetailed Epoch vs Training loss \"iterations\" can be viewed"+ 
      " under \"epochsViewFiles\" directory in \"epochsOutput<datasetName>.csv\" file.\n");
      Console.WriteLine("Please Note 2: \n\tDetailed Epoch vs Training loss \"graph\" can be viewed"+ 
      " under \"epochsViewFiles\" directory in \"TrainingLossWrtNumberOfEpochs.jpeg\" file." +
      "\n\tKindly run the .ipython file for generating the image.\n");
    
    Console.WriteLine("--------------------------------------------------------------------------------------------------------");
    Console.WriteLine();
    Console.WriteLine("Press return/enter key to exit.");
    Console.Read();
  }
}