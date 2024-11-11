using System.Diagnostics.CodeAnalysis;

namespace UCC124111245.ML.Classification;

public static class HelperComputeMetrics {

// ----------------------------------------------------------------------
  /// <summary>
  /// This method computes the accuracy given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>(int, double): Returns tuple #misclassifications and accuracy.</returns>
  public static (int, double) AccuracyAndMisclassifications(
  [DisallowNull] int[,] confusionMatrix)
  {
    int numberOfXis = 0;
    int correctTruePositiveredictions = 0;
    int numberOfClassesInTargetFeaturees = confusionMatrix.GetLength(0);

    for (int i = 0; i < numberOfClassesInTargetFeaturees; i++)
    {
      correctTruePositiveredictions += confusionMatrix[i, i]; // True positives
      for (int j = 0; j < numberOfClassesInTargetFeaturees; j++)
      {
        numberOfXis += confusionMatrix[i, j]; // Total samples
      }
    }

    return 
    (numberOfXis - correctTruePositiveredictions,
    numberOfXis == 0 ? 0 : (double)correctTruePositiveredictions / numberOfXis);
  }

// ----------------------------------------------------------------------

  /// <summary>
  /// This method computes the precsion given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns precion score.</returns>
  public static double Precision(
    [DisallowNull] int[,] confusionMatrix)
  {
    int numberOfClassesInTargetFeaturees = confusionMatrix.GetLength(0);
    double[] precisionScores = new double[numberOfClassesInTargetFeaturees];

    for (int i = 0; i < numberOfClassesInTargetFeaturees; i++)
    {
      int truePositive = confusionMatrix[i, i]; // True positives for class i
      int falsePositive = 0; // False positives for class i

      // Calculate falsePositive for class i
      for (int j = 0; j < numberOfClassesInTargetFeaturees; j++)
      {
        if (j != i)
        {
          falsePositive += confusionMatrix[j, i]; // Column-wise sum for falsePositive
        }
      }

      // Precision = truePositive / (truePositive + falsePositive)
      precisionScores[i] = truePositive + falsePositive == 0 ? 0 : (double)truePositive / (truePositive + falsePositive);
    }

    // Return macro-averaged precision
    return precisionScores.Average();
  }

// ----------------------------------------------------------------------
  
  /// <summary>
  /// This method computes the recall given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns recall score.</returns>
  public static double Recall(
    [DisallowNull] int[,] confusionMatrix)
  {
    int numberOfClassesInTargetFeaturees = confusionMatrix.GetLength(0);
    double[] recallScores = new double[numberOfClassesInTargetFeaturees];

    for (int i = 0; i < numberOfClassesInTargetFeaturees; i++)
    {
      int truePositive = confusionMatrix[i, i]; // True positives for class i
      int falseNegative = 0; // False negatives for class i

      // Calculate falseNegative for class i
      for (int j = 0; j < numberOfClassesInTargetFeaturees; j++)
      {
        if (j != i)
        {
          falseNegative += confusionMatrix[i, j]; // Row-wise sum for falseNegative
        }
      }
      // Recall = truePositive / (truePositive + falseNegative)
      recallScores[i] = truePositive + falseNegative == 0 ? 0 : (double)truePositive / (truePositive + falseNegative);
    }

    // Return macro-averaged recall
    return recallScores.Average();
  }

// ----------------------------------------------------------------------

/// <summary>
  /// This helper method computes the recall for a specific class given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix and class index.</param>
  /// <param name="classIndex">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns recall score for a class.</returns>
  private static double RecallForClass(
    [DisallowNull] int[,] confusionMatrix, 
    [DisallowNull] int classIndex)
  {
    int truePositive = confusionMatrix[classIndex, classIndex];
    int falseNegative = 0;

    for (int j = 0; j < confusionMatrix.GetLength(0); j++)
    {
      if (j != classIndex)
      {
        falseNegative += confusionMatrix[classIndex, j]; // Row-wise sum for falseNegative
      }
    }

    return truePositive + falseNegative == 0 ? 0 : (double)truePositive / (truePositive + falseNegative);
  }

  /// This helper method computes the precison for a specific class given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix and class index.</param>
  /// <param name="classIndex">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns precision score for a class.</returns>
  private static double PrecisionForClass(
    [DisallowNull] int[,] confusionMatrix, 
    [DisallowNull] int classIndex)
  {
    int truePositive = confusionMatrix[classIndex, classIndex];
    int falsePositive = 0;

    for (int j = 0; j < confusionMatrix.GetLength(0); j++)
    {
      if (j != classIndex)
      {
        falsePositive += confusionMatrix[j, classIndex]; // Column-wise sum for falsePositive
      }
    }

    return truePositive + falsePositive == 0 ? 0 : (double)truePositive / (truePositive + falsePositive);
  }


  /// <summary>
  /// This method computes the F1 score given the confusion matrix.
  /// </summary>
  /// <param name="confusionMatrix">This is a non-null and non-empty parameter for confusion matrix.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double: Returns the macro-averaged F1 Score.</returns>
  public static double F1Score(
    [DisallowNull] int[,] confusionMatrix)
  {
    int numberOfClassesInTargetFeaturees = confusionMatrix.GetLength(0);
    double[] f1Scores = new double[numberOfClassesInTargetFeaturees];

    for (int i = 0; i < numberOfClassesInTargetFeaturees; i++)
    {
      double precisionForAClass = PrecisionForClass(confusionMatrix, i);
      double recallForAClass = RecallForClass(confusionMatrix, i);

      // F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
      f1Scores[i] = precisionForAClass + recallForAClass == 0 ? 0 : 2 * (precisionForAClass * recallForAClass) / (precisionForAClass + recallForAClass);
    }

    // Returns macro-averaged F1 Score
    return f1Scores.Average();
  }
// ----------------------------------------------------------------------
}