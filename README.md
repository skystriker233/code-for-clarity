# code-for-clarity
Short description:

The multiband compressor has been implemented successfully. The side chain has been applied. 
It has been tested, the next step is to change the configs of compressors (need more experiments)
It has been combined into enhance.py. The modified enhance.py is also provided. 
Remember to put multibandCompressor.py under the correct dictionary ( same with enhance.py )
Some other scripts like evaluate_remixed are also provided. ( But now it is useless ) 

The next step is to build an intelligent system which can automatically make use of the multiband compressor 
according to the situations of different listeners 

Notice:
Do not creat a band between 250 HZ and 500 HZ to prevent from overflow. 
