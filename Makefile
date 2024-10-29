# Specify the target executable and the source files needed to build it
symnmf: symnmf.o symnmf.h 
	gcc -o symnmf symnmf.o -ansi -Wall -Wextra -Werror -pedantic-errors -lm
# Specify the object files that are generated from the corresponding source files
symnmf.o: symnmf.c
	gcc -c -ansi -Wall -Wextra -Werror -pedantic-errors symnmf.c -lm


