using System.Diagnostics.CodeAnalysis;

namespace UCC124111245.ML.Classification;

public class MultinomialLogisticRegression
{
  private double _learningRate;
  private int _numberOfPredictorFeatures, _numberOfClassesInTargetFeature;
  private double[,] _betas; // betas without betaNots (weights)
  private double[] _betaNots;
  private List<string> _testPredictionsString = new();
  private List<int> _testPredictionsInt = new();
  private int _temperature;

  private int _randomSeed;

  // ----------------------------------------------------------------------
  /// <summary>
  /// This method initialize the parameters (i.e. weights and biases) using Xavier Initialization Method (Glorot Initialization)
  /// </summary>
  /// <remarks>Author: Anish Arya</remarks>
  private void _initWeightsAndBiases()
  {
    // Xavier Initialization Method (Glorot Initialization)
    Random rand = new Random(Seed: this._randomSeed); // for consistency
    double limit = Math.Sqrt(6.0 / (this._numberOfPredictorFeatures + this._numberOfClassesInTargetFeature));
    for (int i = 0; i < this._numberOfPredictorFeatures; i++) 
        for (int j = 0; j < this._numberOfClassesInTargetFeature; j++)
            this._betas[i, j] = rand.NextDouble() * 2 * limit - limit; // range [-limit, limit]

    // Initialize biases to zero
    // this._betaNots = new double[this._numberOfClassesInTargetFeature];
    this._betaNots = this._betaNots.Select(x => 0.0).ToArray();
    return;
  }

  /// <summary>
  /// This parameterised constructor initialises the components/data properties required for the Multinomial Logistic Regression model object to utilize other methods in this class.
  /// </summary>
  /// <param name="numberOfPredictorFeatures">This is a non-null and non-empty parameter for number of input/independent features.</param>
  /// <param name="numberOfClassesInTargetFeature">This is a non-null and non-empty parameter for number of classes in output/target/dependent feature.</param>
  /// <param name="learningRate">This is a non-null and non-empty parameter for learning rate for training.</param>
  /// <param name="randomSeed">This is a non-null and non-empty parameter for consistent results.</param>
  /// <remarks>Author: Anish Arya</remarks>
  public MultinomialLogisticRegression(
    [DisallowNull] int numberOfPredictorFeatures, 
    [DisallowNull] int numberOfClassesInTargetFeature, 
    [DisallowNull] double learningRate,
    [DisallowNull] int randomSeed = 5)
  {
    this._numberOfPredictorFeatures = numberOfPredictorFeatures;
    this._numberOfClassesInTargetFeature = numberOfClassesInTargetFeature;
    this._learningRate = learningRate;
    this._randomSeed = randomSeed;
    
    // Initialize weights and biases
    this._betas = new double[this._numberOfPredictorFeatures, this._numberOfClassesInTargetFeature];
    this._betaNots = new double[this._numberOfClassesInTargetFeature];
    _initWeightsAndBiases();
  }

// ----------------------------------------------------------------------
  /// <summary>
  /// This helper method computes the Ridge regularisation training loss.
  /// </summary>
  /// <param name="mixingParameterLambda">This is a non-null and non-empty parameter for penalizing the model.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns the Ridge(L2) loss.</returns>
  private double _computeL2TrainingLoss(
    [DisallowNull] double mixingParamLambda)
  {
    double l2Loss = 0;
    for (int k = 0; k < this._betas.GetLength(0); k++) // this._betas is the parameter matrix
    {
        for (int j = 0; j < this._betas.GetLength(1); j++)
        {
            l2Loss += this._betas[k, j] * this._betas[k, j]; // Sum of squares of this._betas or weights
        }
    }
    l2Loss = (mixingParamLambda / 2.0) * l2Loss; // Scale by lambda / 2

    return l2Loss;
  }

  /// <summary>
  ///This helper method computes the Cross Entropy(negative log likelihood) loss + Ridge(L2) loss.
  /// </summary>
  /// <param name="X">This is a non-null and non-empty parameter for Input feature training data set.</param>
  /// <param name="yOneHot">This is a non-null and non-empty parameter for one-hot encoded target feature of training set.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns the total loss.</returns>
  private double _computeL2RegularisedEntropyTrainingLoss(
    [DisallowNull] double[,] X, 
    [DisallowNull] double[,] yOneHot)
  {
    double mixingParamLambda = 0.0;
    int numberOfXis = X.GetLength(0);
    int numberOfClassesInTargetFeature = yOneHot.GetLength(1);

    // Initialize total loss
    double trainDataLoss = 0.0, l2Loss = 0.0, totalLoss = 0.0;

    // Compute data loss for each Xi
    for( int i = 0; i < numberOfXis; i++)
    {
      // Get the feature vector for the current sample
      double[] Xi = new double[X.GetLength(1)];
      // Extract the i-th row (Xi)
      Xi = Enumerable.Range(0, X.GetLength(1)).Select(k => X[i, k]).ToArray();
      // use the below "for loop" for large dataset and comment the above line
      // for (int k = 0; k < X.GetLength(1); k++)
      // {
      //   Xi[k] = X[i, k]; // Extract the Xi
      // }

      // Get the Zis for the current Xi
      double[] ZsForGivenXi = this._computeZsForEachTargetClassForGivenXi(Xi);
      // Apply softmax to get probabilities for the current sample
      double[] modifiedProbabilities = this._modifiedSoftmax(ZsForGivenXi, this._temperature);

      // Compute cross-entropy loss for the current Xi
      // trainDataLoss -= Enumerable.Range(0, numberOfClassesInTargetFeature).Sum(
      //   j => yOneHot[i, j] * Math.Log(modifiedProbabilities[j] + 1e-9) // to prevent log(0)
      // );
      for(int j = 0; j < numberOfClassesInTargetFeature; ++j)
      {
        trainDataLoss -= yOneHot[i, j] * Math.Log(modifiedProbabilities[j] + 1e-9);
      }
    }
    // Average the loss over the number of Xis
    trainDataLoss /= numberOfXis;
    
    // Compute l2 loss for Xis
    l2Loss = _computeL2TrainingLoss(
      //numberOfXis, numberOfClassesInTargetFeature, 
      mixingParamLambda);

    totalLoss = trainDataLoss + l2Loss;
    
    return totalLoss;
  }

  /// <summary>
  ///This helper method computes the total loss = Cross Entropy(negative log likelihood) loss + Ridge(L2) loss.
  /// </summary>
  /// <param name="ZsForAGivenXi">This is a non-null and non-empty parameter of Probabilities for each class given a dataset input features' row.</param>
  /// <param name="temperature">This is a non-null and non-empty parameter > 1 for scaling the probabilities to prevent overflow in exponential calculations.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double[]: Returns the modified softmax probabilities.</returns>
  private double[] _modifiedSoftmax(
    [DisallowNull] double[] ZsForAGivenXi,
    [DisallowNull] int temperature = 10)
  {
    // LogSoftmaxWithTemperatureScaling

    // Step 1: Scaling Zs:
    // Subtracting the maximum Zi (divided by temperature) from each Zi improves numerical 
       //stability by preventing large values from causing overflow in exponential calculations.
    double maxZi = ZsForAGivenXi.Max(); 
    // Compute scaled scores by dividing each score by the temperature and 
    double[] exponentsOfTemperatureScaledZs = ZsForAGivenXi.Select(x => Math.Exp((x-maxZi) / temperature)).ToArray();
    // use the below "for loop" for large dataset and comment the above line
    // for (int i = 0; i < ZsForAGivenXi.Length; i++)
    // {
    //     exponentsOfTemperatureScaledZs[i] = Math.Exp((ZsForAGivenXi[i] - maxZi) / temperature);
    // }
    
    // Step 2: Sum-Exp Calculation:
    double sumOfExponentsOfTemperatureScaledZs = exponentsOfTemperatureScaledZs.Sum();
    // use the below "for loop" for large dataset and comment the above line
    // for (int i = 0; i < ZsForAGivenXi.Length; i++)
    // {
    //     sumOfExponentsOfTemperatureScaledZs += Math.Exp(temperatureScaledZs[i]);
    // }

    double[] modifiedSoftMaxOutputProbabilities = new double[this._numberOfClassesInTargetFeature];
    for (int i = 0; i < this._numberOfClassesInTargetFeature; ++i)
    {
      // temperature scaled Exp(Zi)/Sum(temperature scaled Exp(Zi) for all Zs)
      modifiedSoftMaxOutputProbabilities[i] = 
        Math.Exp(ZsForAGivenXi[i] - maxZi) / sumOfExponentsOfTemperatureScaledZs;
    }

    return modifiedSoftMaxOutputProbabilities;
  }

  /// <summary>
  ///This helper method computes the scores for each target class given the dataset input  features' row.
  /// </summary>
  /// <param name="Xi">This is a non-null and non-empty parameter of Input features' row.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double[]: Returns the scores for each target class.</returns>
  private double[] _computeZsForEachTargetClassForGivenXi(
    [DisallowNull] double[] Xi)
  {
    double[] ZsForAGivenXi = new double[this._numberOfClassesInTargetFeature];
    // calculate ZiForAGivenClass for every class and add to the ZsForAGivenXi list
    for(int i = 0; i < this._numberOfClassesInTargetFeature; ++i )
    {
      double ZiForAGivenClass = 0;
      for(int j = 0; j < this._numberOfPredictorFeatures; ++j)
      {
        ZiForAGivenClass += Xi[j] * this._betas[j, i] + this._betaNots[i];
      }
      ZsForAGivenXi[i] = ZiForAGivenClass;
    }
    return ZsForAGivenXi;
  }
  
  /// <summary>
  ///This method trains the dataset using Sequential Gradient Descent.
  /// </summary>
  /// <param name="XTrain">This is a non-null and non-empty parameter of Input features' dataset.</param>
  /// <param name="yOneHotTrain">This is a non-null and non-empty parameter of one-hot encoded Output feature' dataset.</param>
  /// <param name="temperatureForModifiedSoftmax">This is a non-null and non-empty parameter > 1 for scaling the probabilities to prevent overflow in exponential calculations.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns the training loss.</returns>
  public double DTrain(
    [DisallowNull] double[,] XTrain, 
    [DisallowNull] double[,] yOneHotTrain,
    [DisallowNull] int temperatureForModifiedSoftmax = 10)
  {
    double loss = 0.0;
    this._temperature = temperatureForModifiedSoftmax;

    int numberOfXis = XTrain.GetLength(0);
    // Sequential Gradient Descent (no shuffling)
    for (int i = 0; i < numberOfXis; i++)
    {
      // Extract one sample and its one-hot encoded label
      double[] Xi = new double[this._numberOfPredictorFeatures]; // this acts as a vector containing data scores for each predictor feature
      for (int k = 0; k < this._numberOfPredictorFeatures; k++)
        Xi[k] = XTrain[i, k];

      // Forward pass for the sample
      double[] yOneHotXi = new double[this._numberOfClassesInTargetFeature];
      for (int k = 0; k < this._numberOfClassesInTargetFeature; k++)
        yOneHotXi[k] = yOneHotTrain[i, k];

      
      double[] computedZs = this._computeZsForEachTargetClassForGivenXi(Xi);
      double[] modifiedProbabilities = this._modifiedSoftmax(computedZs, this._temperature);

      // Backward pass (compute gradients for one sample)
      double[] error = new double[this._numberOfClassesInTargetFeature];
      for (int j = 0; j < this._numberOfClassesInTargetFeature; j++)
        error[j] = modifiedProbabilities[j] - yOneHotXi[j]; // gradient of loss with respect to softmax output

      // Update weights and biases
      for (int j = 0; j < this._numberOfClassesInTargetFeature; j++)
      {
        for (int k = 0; k < this._numberOfPredictorFeatures; k++)
          this._betas[k, j] -= this._learningRate * error[j] * Xi[k]; // weight update
        this._betaNots[j] -= this._learningRate * error[j]; // bias update
      }
    }
    loss = this._computeL2RegularisedEntropyTrainingLoss(XTrain, yOneHotTrain);

    return loss;
  }

// ----------------------------------------------------------------------

  /// <summary>
  ///This helper method computes the predicted class for a given input row.
  /// </summary>
  /// <param name="Xi">This is a non-null and non-empty parameter of Input features' row.</param>
  /// <param name="classesOfTargetFeature">This is a non-null and non-empty parameter of list classes in Target feature.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns the predicted class.</returns>
  private string _predict(
    [DisallowNull] double[] xi, 
    [DisallowNull] List<string> classesOfTargetFeature)
  {
    // Compute the Zs for the input predictor feature vector
    double[] ZsForAGivenXi = this._computeZsForEachTargetClassForGivenXi(xi);

    // Apply modified softmax to get the probabilities
    double[] modifiedProbabilities = this._modifiedSoftmax(ZsForAGivenXi, this._temperature);

    // Find the index of the class with the highest probability
    string predictedClass = classesOfTargetFeature[Array.IndexOf(modifiedProbabilities, modifiedProbabilities.Max())];

    return predictedClass; // Return the predicted class label
  }

  /// <summary>
  ///This method produces predictions and confusion matrix for test/validation dataset.
  /// </summary>
  /// <param name="XTest">This is a non-null and non-empty parameter of validation dataset of Input features.</param>
  /// <param name="yOneHotTest">This is a non-null and non-empty parameter of validation dataset of one-list classes in Target feature.</param>
  /// <param name="classesOfTargetFeature">This is a non-null and non-empty parameter of list classes in Target feature.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>int[,]: Returns the confusion matrix.</returns>
  public int[,] DTestAndDisplayConfusionMatrix(
    [DisallowNull] double[,] XTest, 
    [DisallowNull] double[,] yOneHotTest,
    [DisallowNull] List<string> classesOfTargetFeature)
  {
    int numberOfXis = XTest.GetLength(0);
    int numberOfClassesInTargetFeature = yOneHotTest.GetLength(1);

    // Initialize the confusion matrix
    int[,] confusionMatrix = new int[numberOfClassesInTargetFeature, numberOfClassesInTargetFeature];

    // Populate confusion matrix by comparing true and predicted classes
    for (int i = 0; i < numberOfXis; i++)
    {
        // Extract sample feature vector
      double[] xi = new double[XTest.GetLength(1)];
      for (int j = 0; j < XTest.GetLength(1); j++)
      {
        xi[j] = XTest[i, j];
      }

      // Determine true class from one-hot encoded label
      int trueClassInt = -1;
      for (int j = 0; j < numberOfClassesInTargetFeature; j++)
      {
        if (yOneHotTest[i, j] == 1.0)
        {
          trueClassInt = j;
          break;
        }
      }

      // Predict the class for xi
      string predictedClassString = this._predict(
                                 xi: xi, classesOfTargetFeature: classesOfTargetFeature);
      this._testPredictionsString.Add(predictedClassString);
      
      int predictedClassInt = classesOfTargetFeature.IndexOf(predictedClassString);
      this._testPredictionsInt.Add(predictedClassInt);
      // Update the confusion matrix at the position [trueClass, predictedClass]
      confusionMatrix[trueClassInt, predictedClassInt]++;
    }
    return confusionMatrix;
  }

// ----------------------------------------------------------------------

  /// <summary>
  ///This method prints the elements of the given confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter of confusionMatrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  public void PrintConfusionMatrix(
    [DisallowNull] int[,] confusionMatrix
  )
  {
    // Print the confusion matrix
    Console.WriteLine("\nConfusion Matrix:");
    Console.Write("       "); // Aligns the top labels
    for (int j = 0; j < this._numberOfClassesInTargetFeature; j++)
    {
      Console.Write($"Pred {j}   ");
    }
    Console.WriteLine();

    for (int i = 0; i < this._numberOfClassesInTargetFeature; i++)
    {
      Console.Write($"True {i}: ");
      for (int j = 0; j < this._numberOfClassesInTargetFeature; j++)
      {
          Console.Write($"{confusionMatrix[i, j],-8} ");
      }
      Console.WriteLine();
    }
  }

// ----------------------------------------------------------------------

}