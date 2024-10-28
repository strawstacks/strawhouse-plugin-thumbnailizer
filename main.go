package main

import (
	"github.com/strawstacks/strawhouse-go"
	"strawhouse-plugin-thumbnailizer/plugin"
)

func Plugin() strawhouse.Plugin {
	return new(plugin.Plugin)
}
