#!/bin/bash
echo "Zipping..."
zip -u wgt.zip *.wgt
echo "Cleaning..."
rm *.wgt
echo "Done!"

