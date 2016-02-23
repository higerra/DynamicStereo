#include "configurator.h"

std::map<std::string, int> Configurator::config_int;
std::map<std::string, bool> Configurator::config_bool;
std::map<std::string, double> Configurator::config_double;

void Configurator::printConfig(){
    for(std::map<std::string, int>::const_iterator iter = config_int.begin(); iter!=config_int.end(); ++iter)
	printf("%s: %d\n", iter->first.c_str(), iter->second);
    for(std::map<std::string, double>::const_iterator iter = config_double.begin(); iter!=config_double.end(); ++iter)
	printf("%s: %.2f\n", iter->first.c_str(), iter->second);
    for(std::map<std::string, bool>::const_iterator iter = config_bool.begin(); iter!=config_bool.end(); ++iter)
	printf("%s: %d\n", iter->first.c_str(), iter->second);
}
