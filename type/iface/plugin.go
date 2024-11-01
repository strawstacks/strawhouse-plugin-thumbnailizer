package iface

import "github.com/strawstacks/strawhouse-go"

type Callbacker interface {
	Callback() strawhouse.PluginCallback
}
