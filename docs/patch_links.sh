#!/bin/sh

find _build -name '*.html' | xargs -r sed -f post.sed -i
