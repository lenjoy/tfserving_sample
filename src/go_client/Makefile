GOCMD=go
GOPATH=$(dir $(abspath ..))
OUTPUT_DIR=$(GOPATH)bin


.PHONY: go_client
go_client:
	@echo "GOPATH=$(GOPATH)"
	cd $(GOPATH)
	mkdir -p $(OUTPUT_DIR)/go_client

	@GOPATH=$(GOPATH) $(GOCMD) build -v -o $(OUTPUT_DIR)/go_client/client go_client/cmd/client/$(wildcard *.go)
	@echo $(OUTPUT_DIR)/go_client/client

	@GOPATH=$(GOPATH) $(GOCMD) build -v -o $(OUTPUT_DIR)/go_client/client2 go_client/cmd/client2/$(wildcard *.go)
	@echo $(OUTPUT_DIR)/go_client/client2


clean:
	rm -rf $(OUTPUT_DIR)
