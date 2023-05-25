#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good(); // Restituisce true se il file esiste e può essere aperto correttamente
}

int load_data( std::string filename, 
                std::vector<float> &x1,
                std::vector<float> &x2,
                std::vector<float> &x3,
                std::vector<float> &y) {


    if(fileExists(filename)== false){
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

        // std::sscanf(line.c_str(), "%f,%f,%f,%f", &x1_val, &x2_val, &x3_val, &y_val);
        // std::cout << "data: " << row[0] << "," << row[1] << "," <<row[2] << " " <<std::endl;
        // x3_val << " " << y_val << " " << "\n";
        x1.push_back(row[0]);
        x2.push_back(row[1]);
        x3.push_back(row[2]);
        y.push_back(row[3]);
        i++;
    }
  return i;

}
