#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

bool fileExists(const std::string &filePath) {
  std::ifstream file(filePath);
  return file.good(); // Restituisce true se il file esiste e può essere aperto
                      // correttamente
}

int load_data(std::string filename, std::vector<float> &x1,
              std::vector<float> &x2, std::vector<float> &x3,
              std::vector<float> &y) {

  if (fileExists(filename) == false) {
    return -1;
  }

  // Open file
  std::ifstream file(filename);

  // Read header
  std::string line;
  std::getline(file, line);

  // Read data
  int i = 0;
  while (std::getline(file, line)) {

    std::stringstream ss(line);
    std::vector<double> row;
    std::string value;

    // Leggi ogni valore separato da virgola e salvalo nell'array
    while (std::getline(ss, value, ',')) {
      double number = std::stod(value);
      row.push_back(number);
    }

    x1.push_back(row[0]);
    x2.push_back(row[1]);
    x3.push_back(row[2]);
    y.push_back(row[3]);
    i++;
  }
  return i;
}
