#ifndef UPDATOR__H
#define UPDATOR__H

#include <string>
namespace Updator {
	void InitializeWaitbar(const std::string& str);
	void UpdateWaitbar(float nValue);
	void CloseWaitbar();
}

#endif
