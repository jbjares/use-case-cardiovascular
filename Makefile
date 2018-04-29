.PHONY: build

IMAGE_NAME = research:cardiovascular

build:
	docker build --rm --no-cache -t $(IMAGE_NAME) .

