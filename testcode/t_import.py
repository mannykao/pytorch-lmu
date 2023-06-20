from pathlib import Path
from mkpyutils.importutils import import1, importFiles, importFolder, importAllPackages

if __name__ == "__main__":
	pkgname = 'lmu.fftlmu'

	modules =[
#		'lmu',
		'lmu',
		'lmufft',
		'lmuapp',
	]
	#1: test importing one module using an explicit path
	for module in modules:
		print(import1(pkgname, module, tag=" "))

#	importFiles('lmu', modules)

	#2: import using find_packages() - simulate what setup.py will put into the package	
	srcroot = Path(__file__).parent.parent
	module = importFolder(pkgname, srcroot)
	print(f"{module=}")

	print(f"{srcroot=}")
	imported = importAllPackages(where=srcroot, srcroot=srcroot, logging=False)

	for folder in imported:
		for imp1 in folder:
			if type(imp1) is not Error:
				print(f"imported {imp1.__name__} as {imp1.__package__}")
			else:
				print(f" '{imp1.errmsg}'")	
