.SUFFIXES: .c .u
CC= gcc
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS -g -Wall -O3 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF -Winline -fast -I/opt/local/include/gsl
CFLAGS = -g -Wall
LDFLAGS=  -lgsl -lgslcblas

LOBJECTS=  main.o typedefs.o utils.o em.o inference.o model.o corpus.o

LSOURCE=  main.c typedefs.c utils.c em.c inference.c model.c corpus.c

main:	$(LOBJECTS)
	$(CC) $(CFLAGS) $(LOBJECTS) -o slda $(LDFLAGS)

clean:
	-rm -f *.o *.so

so:
	R CMD SHLIB r-interface.c em.c inference.c utils.c model.c corpus.c typedefs.c;
