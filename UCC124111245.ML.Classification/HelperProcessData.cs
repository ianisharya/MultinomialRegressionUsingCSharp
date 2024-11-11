using System.Diagnostics.CodeAnalysis;

namespace UCC124111245.ML.Classification;

public static class HelperProcessData
{
  /// <summary>
  /// This method seprates one-hot encodes the classes in the the target feature data.
  /// </summary>
  /// <param name="numDataXis">This is a non-null and non-empty parameter number of rows in the dataset.</param>
  /// <param name="numClassesInFeature">This is a non-null and non-empty parameter of number of classes in the feature.</param>
  /// <param name="y">This is a non-null and non-empty parameter of target class.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double[]: Returns one-hot encoded y.</returns>
  public static double[,] OneHotEncode(
    [DisallowNull] int numDataXis,
    [DisallowNull] int numClassesInFeature, 
    [DisallowNull]int[] y)
  {
    double[,]? yOneHot;    
    yOneHot = new double[numDataXis, numClassesInFeature];
    for (int rowidx = 0; rowidx < numDataXis; ++rowidx)
    {
#pragma warning disable CS8602 // Dereference of a possibly null reference.
      // save 1 at position of (index of class labels of target features)
      yOneHot[rowidx, y[rowidx]] = 1.0;
#pragma warning restore CS8602 // Dereference of a possibly null reference.
    }
    return yOneHot;
  }
// ----------------------------------------------------------------------

  /// <summary>
  /// This helper method gets the lebel-index from the list of classes in Target feature.
  /// </summary>
  /// <param name="classesInTargetFeature">This is a non-null and non-empty parameter of number of classes in the target feature.</param>
  /// <param name="classLabel">This is a non-null and non-empty string parameter of class to be label-encoded</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>int: Returns lebel of the class.</returns>
  private static int _getClassIndexFromClassesInTargetFeature(
    [DisallowNull] List<string> classesInTargetFeature, // to maintain the order of encodedness
    [DisallowNull] string classLabel) {
    // to label-encode the target feature
    return classesInTargetFeature.IndexOf(classLabel);
  }

  /// <summary>
  /// This helper method checks whether the class already exists in the classes list
  /// </summary>
  /// <param name="classesInTargetFeature">This is a non-null and non-empty parameter number of classes in the target feature.</param>
  /// <param name="classLabel">This is a non-null and non-empty string parameter of class to be label-encoded</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>bool: Returns true if the class exists in the list.</returns>
  private static bool _doesClassLabelAlreadyExistsInList(
    [DisallowNull] List<string> classesInTargetFeature, 
    [DisallowNull] string classLabel) {
    return classesInTargetFeature.Contains(classLabel);
  }

  /// <summary>
  /// This method seprates The input features data and the target feature data.
  /// </summary>
  /// <param name="X">This is a non-null and non-empty parameter of input feature dataset.</param>
  /// <param name="y">This is a non-null and non-empty parameter of target feature dataset.</param>
  /// <param name="datasetFile">This is a unitialised yet non-null and non-empty parameter of dataset file which is already read.</param>
  /// <param name="numDataXis">This is a non-null and non-empty parameter of numbe of rows in the dataset.</param>
  /// <param name="numTotalFeatures">This is a non-null and non-empty parameter of number of total input features.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>(double[,], int[], List<string>): Returns tuple of X, y, and classes in target feature.</returns>
  public static (double[,], int[], List<string>) GetXyPairs( 
    [DisallowNull] double[,] X, 
    [DisallowNull] int[] y, 
    [DisallowNull] List<string> datasetFile,
    [DisallowNull] int numDataXis,
    [DisallowNull] int numTotalFeatures
    )
  {
    X = new double[numDataXis, numTotalFeatures-1];
    y = new int[numDataXis];
    
    // to store the classes present in the target feature and use the index of the labels in the list for label encoding 
    // to feed it to convert to one-hot encoder
    List<string> classesInTargetFeature = new();

    // to keep the track of row and column numbers | to be used to seprate X's and y's
    int rowidx = 0, colidx = 0; // initialize xi(row number) and score(or data observation) number
    string[] scores; // to keep track of scores at every xi

    foreach(string xi in datasetFile)
    {
      scores = xi.Split(",");
      // for each scores, store it in X and y as per data and target feature
      foreach(string score in scores)
      {
        if( colidx == numTotalFeatures-1) // assuming target class feature is the last column
        {
          // fetch the index from the Classes list if class already exists
          // else add and then fetch the index
          if(!_doesClassLabelAlreadyExistsInList(classesInTargetFeature, score))
          {
            // add the label, if not exists
            classesInTargetFeature.Add(score);
          } // end if
          #pragma warning disable CS8602 // Dereference of a possibly null reference.
          // fetch the label index and add it to "y"
          y[rowidx] = _getClassIndexFromClassesInTargetFeature(classesInTargetFeature, score);
        } // end if
        else
        {
          // add it to x as it is not a target feature
          X[rowidx, colidx] = double.Parse(score);
          #pragma warning restore CS8602 // Dereference of a possibly null reference.
        }
        ++colidx;
      } // end foreach: score in scores
      colidx = 0; // reset column index
      ++rowidx; // move to next xi
    } // end foreach: xi in dataset
    return (X, y, classesInTargetFeature);
  }


// ----------------------------------------------------------------------
  /// <summary>
  /// This helper method converts a list of arrays into a 2-d array .
  /// </summary>
  /// <param name="listOfArrays">This is a non-null and non-empty parameter of list of arrays.</param>
  /// <param name="numberOfColumns">This is a non-null and non-empty parameter of number of columns of the arrays in the list.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double[,]: Returns 2-d array.</returns>
  private static double[,] _getArray2D(
    [DisallowNull] List<double[]> listOfArrays, 
    [DisallowNull] int numberOfColumns)
  {
    double[,] array2D = new double[listOfArrays.Count, numberOfColumns];
    for (int i = 0; i < listOfArrays.Count; ++i)
    {
      for (int j = 0; j < numberOfColumns; ++j)
      {
        array2D[i, j] = listOfArrays[i][j];
      }
    }
    return array2D;
  }

  /// <summary>
  /// This helper method extracts a row from a nd-array given the index.
  /// </summary>
  /// <param name="ndarray">This is a non-null and non-empty parameter of n dimensional array</param>
  /// <param name="rowIndex">This is a non-null and non-empty parameter of index of row nd-array.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>double[]: Returns the row of the nd-array.</returns>
  private static double[] _getxi(
    [DisallowNull] double[,] ndarray, 
    [DisallowNull] int rowIndex)
  {
    double[] xi = new double[ndarray.GetLength(1)];
    for (int i = 0; i < xi.Length; i++)
    {
      xi[i] = ndarray[rowIndex, i];
    }
    return xi;
  }

  /// <summary>
  /// This method gives the stratefied splits of traininng and validation datasets.
  /// </summary>
  /// <param name="X">This is a non-null and non-empty parameter of input feature dataset to be split.</param>
  /// <param name="y">This is a non-null and non-empty parameter of one-hot encoded target feature dataset to be split.</param>
  /// <param name="randomSeed">This is a non-null and non-empty parameter of random seed for consistent results.</param>
  /// <param name="testSize">This is a non-null and non-empty parameter of test dataset siz (between 0 and 1).</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>(double[,], double[,], double[,], double[,]): Returns the Stratified Train-Test Split tuple of (XTrain, yTrain, XTest, yTest)</returns>
  public static (double[,], double[,], double[,], double[,]) StratifiedTrainTestSplit(
    [DisallowNull] double[,] X, 
    [DisallowNull] double[,] yOneHot, 
    [DisallowNull] int randomSeed,
    [DisallowNull] double testSize)
    {
      if(randomSeed < 1)
      {
        randomSeed = 5;
      }
      int numberOfXis = X.GetLength(0);
      int numberOfPredictorFeatures = X.GetLength(1);
      int numberOfClassesInTargetFeature = yOneHot.GetLength(1);
      
      // Lists to contain both the training as well as test Xis
      List<double[]> XTrain = new();
      List<double[]> XTest = new();
      List<double[]> yTrain = new();
      List<double[]> yTest = new();

      Random random = new Random(Seed: randomSeed);

      // Separate xi by class to ensure stratified split
      for (int k = 0; k < numberOfClassesInTargetFeature; ++k)
      {
        List<int> classXis = Enumerable.Range(0, numberOfXis)
                                            .Where(i => yOneHot[i, k] == 1.0)
                                            .ToList();
        // Shuffle samples to add randomness
        List<int> shuffledClassXis = classXis.OrderBy(x => random.Next()).ToList();

        // Split the samples based on the testSize
        int numTestSamples = (int)(shuffledClassXis.Count * testSize);
        var testIndices = shuffledClassXis.Take(numTestSamples).ToList();
        var trainIndices = shuffledClassXis.Skip(numTestSamples).ToList();

        // Adding xi to training and test data
        foreach (int idx in trainIndices)
        {
          XTrain.Add(_getxi(X, idx));
          yTrain.Add(_getxi(yOneHot, idx));
        }
        foreach (int idx in testIndices)
        {
          XTest.Add(_getxi(X, idx));
          yTest.Add(_getxi(yOneHot, idx));
        }
        // Convert lists to 2D arrays and return tuple for final output
      }
      return (_getArray2D(XTrain, numberOfPredictorFeatures), // XTrain
                _getArray2D(yTrain, numberOfClassesInTargetFeature),// yOneHotTrain
                _getArray2D(XTest, numberOfPredictorFeatures), // XTest
                _getArray2D(yTest, numberOfClassesInTargetFeature)); // yOneHotTest
    }
}