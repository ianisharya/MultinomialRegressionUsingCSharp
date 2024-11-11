using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace UCC124111245.Utilities;

public static class HelperDirectoryMethods
{
  /// <summary>
  /// This helper method checks if the directory exists in the system or not.
  /// </summary>
  /// <param name="directoryPath">This is a non-null and non-empty parameter.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>bool: Returns true if the given directory exists else returns false.</returns>
  public static bool IsDirectoryExists([DisallowNull] DirectoryInfo directoryPath)
  {
    return Directory.Exists(directoryPath.ToString());
  } // end: method "IsDirectoryExists"

  /// <summary>
  /// This helper method checks if the file exists in the system or not. 
  /// </summary>
  /// <param name="filePath">This is a non-null and non-empty parameter.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>bool: Returns true if the given file exists else returns false.</returns>
  public static bool IsFileExists([DisallowNull] StringBuilder filePath)
  {
    return File.Exists(filePath.ToString());
  } // end: method "IsFileExists"
  
  /// <summary>
  /// This helper method checks the availability of the child directory 
  /// and if available, fetches the parent directory.
  /// </summary>
  /// <param name="childDirectory">This is a non-null and non-empty parameter.</param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>Directoryinfo?: Returns the Parent Directory if the parent directory exists 
  /// else returns null in case of:
  /// a. The child directory is a root, or
  /// b. The child directory does not exist.</returns>
  public static DirectoryInfo? GetParentDirectory(
    [DisallowNull] DirectoryInfo childDirectory)
  {
    DirectoryInfo? parentDirectory = null; 
    // should not be a root
    if ((childDirectory.ToString() != @"/") && (IsDirectoryExists(childDirectory))) {
      parentDirectory = Directory.GetParent(childDirectory.ToString());
    }
    return parentDirectory;
  }

  /// <summary>
  /// This helper method fetches the current directory of the Debug/Release file.
  /// </summary>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>DirectoryInfo: Returns the current directory of the Debug/Release file.</returns>
  public static DirectoryInfo GetCurrentDirectory()
  {
    return new DirectoryInfo(Directory.GetCurrentDirectory());
  }
  
  /// <summary>
  /// This helper method creates the directory at the specified path if it does not exist.
  /// It firsts check that whether parent directory exists or not before creation.
  /// </summary>
  /// <param name="directoryPath"></param>
  /// <remarks>Author: Anish Arya</remarks>
  /// <returns>bool: returns true if the parent directory exists else returns false.</return>
  public static bool CreateDirectory([DisallowNull] DirectoryInfo directoryPath)
  {
    bool isCreate = false;
    if(GetParentDirectory(directoryPath) is null)
    {
      return isCreate;
    }
    Directory.CreateDirectory(directoryPath.ToString());
    isCreate = true;
    return isCreate;
  }

} // end: class "DirectoryMethods"