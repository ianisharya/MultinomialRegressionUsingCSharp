#pragma warning disable IDE1006 // Naming Styles

using System.Diagnostics.CodeAnalysis;
using System.Text;
using UCC124111245.ML.Classification;


namespace UCC124111245.Utilities;

/// <summary>
/// This Class reads the CSV and is able to seperate input features and traget class feature from the dataset.
/// </summary>
/// <exception cref="FileNotFoundException">File not found for CSV reader.</exception>
/// <exception cref="InvalidDataException">Input File format should be of type .csv.</exception>
/// <exception cref="FileNotFoundException">No data in Input File.</exception>
/// <remarks>Author: Anish Arya</remarks>
public class ClassificationCSVReader
{
  public double[,]? X { get; set;} // input features
  public int[]? y { get; set;} // lebel-encoded classes of target feature

  public int NumDataXis { get; set;} // number of rows in the dataset
  public int NumTotalFeatures { get; set;}

  private string? DataFilePath;
  public List<string> DatasetFile = new();

  public List<string> ClassesOfTargetFeature = new();

  /// <summary>
  /// This method reads all the rows in the dataset.
  /// </summary>
  /// <param name="dataFilePath">This is a non-null and non-empty parameter for data file' path..</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <exception cref="FileNotFoundException">File not found for CSV reader.</exception>
  /// <exception cref="InvalidDataException">Input File format should be of type .csv.</exception>
  /// <exception cref="FileNotFoundException">No data in Input File.</exception>
  public void ReadCSVData([DisallowNull] string dataFilePath)
  {  
    /* Checks for file validity */
    // if file does not exist, throw FileNotFoundException
    if(!HelperDirectoryMethods.IsFileExists(new StringBuilder(dataFilePath)))
    {
        // file not found
        throw new FileNotFoundException("File not found for CSV reader.");
    }
    string[] filenameSplits = dataFilePath.Split("." );

    if (filenameSplits[filenameSplits.Length - 1] != "csv") 
    {
        // must be a csv file
        throw new InvalidDataException("Input File format should be of type .csv.");
    }
    DataFilePath = dataFilePath;

    // read all the lines and store it line by line in list of strings
#pragma warning disable CS8604 // Possible null reference argument.
    DatasetFile = File.ReadAllLines(DataFilePath, Encoding.UTF8).ToList<string>();
#pragma warning restore CS8604 // Possible null reference argument.
    NumDataXis = DatasetFile.Capacity; // number of rows
    // if file is empty
    if (NumDataXis <= 0) throw new InvalidDataException("No data in Input File.");
    // get the number of datapoints in the first row of the csv file to get the num of total features
    NumTotalFeatures = DatasetFile[0].Split(",").Length;
    return; // data read successfully
  } // end method: ReadCSVData

// --------------------------------------------------------------------

  /// <summary>
  /// This method seprates The input features data and the target feature data.
  /// Assumption: The target class feature must be last column.
  /// </summary>
  /// <param name="dataFilePath">This is a non-null and non-empty parameter for data file' path..</param>
  /// <remarks>Author: Anish Arya</remarks>
  public void ProcessXYPairs()
  {
    // get X and y from dataset
    var XYClassesTrio = HelperProcessData.GetXyPairs(
      new double[ this.NumDataXis, this.NumTotalFeatures-1], 
      new int[this.NumDataXis], 
      this.DatasetFile, 
      this.NumDataXis, 
      this.NumTotalFeatures);
    
    this.X = XYClassesTrio.Item1;
    this.y = XYClassesTrio.Item2;
    this.ClassesOfTargetFeature = XYClassesTrio.Item3;

    return; // processed X and Y successfully
  } // end method: ProcessXYPairs

} // end class: ClassificationCSVReader