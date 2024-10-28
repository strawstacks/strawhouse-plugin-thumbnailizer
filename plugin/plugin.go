package plugin

import "github.com/strawstacks/strawhouse-go"

type Plugin struct {
	callback strawhouse.PluginCallback
}

func (r *Plugin) Load(callback strawhouse.PluginCallback) {
	r.callback = callback
}

func (r *Plugin) Unload() {
	r.callback = nil
}
