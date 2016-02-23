#ifndef CONFIGURATOR_H
#define CONFIGURATOR_H
#include <map>
#include <string>
#include <exception>
#include <fstream>
#include <typeinfo>

class Configurator{
public:
    template<typename T>
	static void setValue(const std::string& name, T value);
    
    template<typename T>
	static T getValue(const std::string& name);

    static void printConfig();
private:
    template<typename T>
	static T retriveValue(const std::map<std::string, T>& m, const std::string& name);
    static std::map<std::string, int> config_int;
    static std::map<std::string, bool> config_bool;
    static std::map<std::string, double> config_double;
};

template<typename T>
T Configurator::retriveValue(const std::map<std::string, T>& m, const std::string& name){
    if(m.find(name) == m.end())
	throw std::runtime_error("Configurator::getValue(): param not found, name=" + name);
    return m.at(name);
}

template<typename T>
void Configurator::setValue(const std::string& name, T value){
    if(typeid(value).hash_code() == typeid(int).hash_code())
	config_int[name] = value;
    else if(typeid(value).hash_code() == typeid(double).hash_code())
	config_double[name] = value;
    else if(typeid(value).hash_code() == typeid(bool).hash_code())
	config_bool[name] = value;
    else
	throw std::runtime_error("Configurator::setValue(): type not suppoert!");
}

template<typename T>
T Configurator::getValue(const std::string& name){
    T temp;
    if(typeid(temp).hash_code() == typeid(int).hash_code())
	return retriveValue<int>(config_int, name);
    else if(typeid(temp).hash_code() == typeid(double).hash_code())
	return retriveValue<double>(config_double, name);
    else if(typeid(temp).hash_code() == typeid(bool).hash_code())
	return retriveValue<bool>(config_bool, name);
    else
	throw std::runtime_error("Configurator::getValue(): type not supported!");
}


#endif
